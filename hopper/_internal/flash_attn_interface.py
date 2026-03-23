# Copyright (c) 2023, Tri Dao.

from typing import Optional, Tuple, Union

import os
import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import back_lite._C # Registers operators with PyTorch

# isort: on

flash_attn_3_cuda = torch.ops.back_lite

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

# Fused Triton kernel: single-pass mask generation from cumulative LSE.
from back_lite._internal.tile_stats_reduce import mask_from_stats_fused as _mask_from_stats_fused


def _mask_from_stats_compiled(tile_lse, softmax_lse, num_row_tiles, kBlockM_bwd, negl_prob):
    """Mask generation from cumulative LSE.  Delegates to fused Triton kernel."""
    return _mask_from_stats_fused(tile_lse, softmax_lse, num_row_tiles, kBlockM_bwd, negl_prob)


def _generate_mask_from_stats_mass(tile_stats, seq_q, head_dim, head_dim_v, dtype, negl_prob, softmax_lse, is_causal=False, is_local=False, softcap=0.0):
    """
    Generate block sparsity mask from forward per-row tile LSE statistics for backward kernel.

    tile_stats: [batch, head, seq_q, N] float32 — per-block LSE (log2 domain, -inf for masked).
    softmax_lse: [batch, head, seq_q] float32 — full-row LSE (ln domain) from FA3 forward.
    N is relative for local window, logical for full/causal attention.
    """
    if tile_stats.dim() != 4: raise ValueError("tile_stats must have shape [batch, head, seq_q, num_n_blocks_fwd]")
    batch, head, actual_seq_q, num_n_blocks_fwd = tile_stats.shape

    has_softcap = softcap > 0.0
    element_size = 1 if dtype == torch.int8 else (2 if dtype in (torch.float16, torch.bfloat16) else 4)
    kBlockM_bwd, kBlockN_bwd = flash_attn_3_cuda.get_tile_size_bwd(head_dim, head_dim_v, is_causal, is_local, has_softcap)
    tile_info_fwd = flash_attn_3_cuda.get_tile_size_fwd_sm90(
        head_dim, head_dim_v, is_causal, is_local,
        element_size, False, False, has_softcap, dtype == torch.int8,
    )
    kBlockN_fwd = tile_info_fwd[1]

    num_row_tiles_bwd = (seq_q + kBlockM_bwd - 1) // kBlockM_bwd

    if softmax_lse is None: raise RuntimeError("Mask generation requires softmax_lse.")
    if kBlockN_bwd != kBlockN_fwd: raise RuntimeError(f"Mask generation requires matching N tile sizes (bwd={kBlockN_bwd}, fwd={kBlockN_fwd}).")
    if seq_q != actual_seq_q: raise RuntimeError(f"Mask generation requires seq_q == tile_stats.shape[2] (seq_q={seq_q}, tile_stats_seq_q={actual_seq_q}).")

    mask = _mask_from_stats_fused(tile_stats, softmax_lse, num_row_tiles_bwd, kBlockM_bwd, negl_prob).contiguous()

    if os.environ.get('DEBUG_BACKLITE_SPARSITY'):
        # Reconstruct uint8[B, H, Tm, N] view from the int32 bitmask for sparsity stats
        B_m, H_m, N_m, num_words = mask.shape
        mask_u8 = torch.zeros(B_m, H_m, num_row_tiles_bwd, N_m, dtype=torch.uint8, device=mask.device)
        for m in range(num_row_tiles_bwd):
            w, bit = m // 32, m % 32
            mask_u8[:, :, m, :] = ((mask[..., w] >> bit) & 1).to(torch.uint8)
        stats = compute_sparsity(tile_stats, mask_u8, seq_q, is_causal=is_causal)
        mode = "causal" if is_causal else ("local" if is_local else "full")
        print(f"[BackLite] mode={mode}  overall_sparsity={stats['causal_sparsity']:.2%}  additional_sparsity={stats['additional_sparsity']:.2%}")

    return mask


def compute_sparsity(tile_stats: torch.Tensor, mask: torch.Tensor, seq_q: int, is_causal: bool = True) -> dict:
    """Compute backward sparsity metrics from forward tile statistics and backward block mask.

    Two complementary metrics:

    - ``causal_sparsity``: fraction of *relevant* tiles skipped in the backward pass.
      When ``is_causal=True``: denominator = causal triangle tiles (lower triangle).
      When ``is_causal=False``: denominator = all tiles (full attention matrix).

    - ``additional_sparsity``: fraction of *actually-planned* tiles skipped, where
      "planned" means the forward pass touched that tile (``tile_stats > -inf``).
      For full-attention (L) layers this equals ``causal_sparsity``; for sliding-window
      (S) layers the window already excludes many causal tiles, so this metric shows only
      BackLite's extra savings on top of both causal *and* window masking.

    Args:
        tile_stats: Forward tile LSE tensor, shape ``[B, H, seq_q, N_k]``.  Values are
            in log2 domain; ``-inf`` for tiles outside causal/window attention.
        mask: Backward block sparsity mask from :func:`_generate_mask_from_stats_mass`,
            shape ``[B, H, Tm, N_k]`` (bool or float).  ``True``/1 = tile computed.
        seq_q: Query sequence length (= ``tile_stats.shape[2]``).
        is_causal: If True, first metric uses causal triangle; if False, uses full matrix.

    Returns:
        ``dict`` with float keys ``'causal_sparsity'`` and ``'additional_sparsity'``.
    """
    B, H, Tm, N_k = mask.shape
    kBlockM = seq_q // Tm   # query rows per backward tile
    kBlockN = seq_q // N_k  # key cols per block (assumes square, same seq_k)

    # --- region sparsity: causal triangle (causal) or full matrix (noncausal) ---
    if is_causal:
        m_idx = torch.arange(Tm, device=mask.device)
        n_idx = torch.arange(N_k, device=mask.device)
        causal_active = (n_idx[None, :] * kBlockN) < ((m_idx[:, None] + 1) * kBlockM)
        n_region = causal_active.sum()
        n_computed = mask.float()[:, :, causal_active].sum()
    else:
        n_region = Tm * N_k
        n_computed = mask.float().sum()
    region_density = n_computed / max(float(n_region * B * H), 1.0)

    # --- additional sparsity (within actually-planned tiles: causal + window) ---
    # Aggregate kBlockM forward rows into each backward tile row via max pooling.
    ts_tiled = tile_stats.reshape(B, H, Tm, kBlockM, N_k).amax(dim=3)  # [B, H, Tm, N_k]
    planned = ts_tiled > -1e37  # True for tiles the forward actually attended to
    n_planned = planned.sum().float().clamp(min=1)
    n_bwd = mask.float().sum()
    additional_density = n_bwd / n_planned

    return {
        'causal_sparsity': (1.0 - region_density).clamp(0.0, 1.0).item(),
        'additional_sparsity': (1.0 - additional_density).clamp(0.0, 1.0).item(),
    }


def _flash_attn_forward(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=True,
        scheduler_metadata=None,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
        tile_stats=None,
    ):
    q, k, k_new, v_new = [maybe_contiguous(x) for x in (q, k, k_new, v_new)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
    ]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table, kv_batch_idx, leftpad_k = [
        maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    seqlens_rotary = maybe_contiguous(seqlens_rotary)
    out, softmax_lse, *rest = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
        tile_stats=tile_stats,
    )
    return out, softmax_lse, *rest


def _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
        block_mask=None,
):
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, *rest = flash_attn_3_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        sm_margin,
        block_mask=block_mask,
    )
    return dq, dk, dv, softmax_d


# ---------------------------------------------------------------------------
# torch.compile-compatible custom ops
# These replace FlashAttnFunc (torch.autograd.Function) for the flash_attn_func
# path so that torch.compile can include them in compiled graphs without
# triggering graph breaks or recompilations.
# ---------------------------------------------------------------------------

@torch.library.custom_op("back_lite::_flash_attn_fwd", mutates_args=(), device_types="cuda")
def _back_lite_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    num_splits: int,
    deterministic: bool,
    sm_margin: int,
    negl_prob: float,
    block_n_size: int,
    block_m_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (out, softmax_lse, tile_stats).
    tile_stats has shape (B, H, T, N_k) when negl_prob > 0, else shape (0,) empty placeholder.
    N_k = N_local (reduced) when is_local, else N_full.
    """
    _sparse = negl_prob > 0.0
    if _sparse:
        batch, seq_q, heads, head_dim = q.shape
        is_local = window_size_left >= 0 or (not causal and window_size_right >= 0)
        if is_local:
            wl = max(window_size_left, 0)
            wr = 0 if causal else max(window_size_right, 0)
            num_n_blocks = (wl + wr + block_m_size + block_n_size - 1) // block_n_size + 1
        else:
            num_n_blocks = (k.shape[1] + block_n_size - 1) // block_n_size
        seq_q_pad = (seq_q + block_m_size - 1) // block_m_size * block_m_size
        tile_stats: Optional[torch.Tensor] = torch.full(
            (batch, heads, num_n_blocks, seq_q_pad), float('-inf'),
            device=q.device, dtype=torch.float32,
        ).permute(0, 1, 3, 2)  # [B,H,seq_q_pad,N] stride_m=1 (non-contiguous is intentional)
        # NOTE: do NOT call .contiguous() here – permute gives stride_m=1 which the
        # FA3 forward kernel requires when writing tile_stats.  Calling .contiguous()
        # would relayout to stride_m=N, causing cudaErrorMisalignedAddress in the kernel.
    else:
        tile_stats = None

    out, softmax_lse, *_ = _flash_attn_forward(
        q, k, v,
        None, None,        # k_new, v_new
        None,              # qv
        None,              # out
        None, None, None,  # cu_seqlens_q/k/k_new
        None, None,        # seqused_q/k
        None, None,        # max_seqlen_q/k
        None, None, None,  # page_table, kv_batch_idx, leftpad_k
        None, None, None,  # rotary_cos/sin, seqlens_rotary
        None, None, None,  # q/k/v_descale
        softmax_scale,
        causal=causal,
        window_size=(window_size_left, window_size_right),
        softcap=softcap,
        num_splits=num_splits,
        sm_margin=sm_margin,
        tile_stats=tile_stats,
    )

    if _sparse:
        # Slice back to actual seq_q rows (discarding the padding rows)
        return out, softmax_lse, tile_stats[:, :, :seq_q, :]
    else:
        return out, softmax_lse, q.new_empty(0)


@torch.library.register_fake("back_lite::_flash_attn_fwd")
def _back_lite_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    num_splits: int,
    deterministic: bool,
    sm_margin: int,
    negl_prob: float,
    block_n_size: int,
    block_m_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, D = q.shape
    Dv = v.shape[-1]
    out = torch.empty((B, T, H, Dv), dtype=q.dtype, device=q.device)
    softmax_lse = torch.empty((B, H, T), dtype=torch.float32, device=q.device)
    if negl_prob > 0.0:
        is_local = window_size_left >= 0 or (not causal and window_size_right >= 0)
        if is_local:
            wl = max(window_size_left, 0)
            wr = 0 if causal else max(window_size_right, 0)
            Nk = (wl + wr + block_m_size + block_n_size - 1) // block_n_size + 1
        else:
            Nk = (T + block_n_size - 1) // block_n_size
        T_pad = (T + block_m_size - 1) // block_m_size * block_m_size
        tile_stats = torch.empty((B, H, Nk, T_pad), dtype=torch.float32, device=q.device).permute(0, 1, 3, 2)[:, :, :T, :]
    else:
        tile_stats = q.new_empty(0)
    return out, softmax_lse, tile_stats


@torch.library.custom_op("back_lite::_flash_attn_bwd", mutates_args=(), device_types="cuda")
def _back_lite_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    tile_stats: torch.Tensor,  # (B,H,T,Nk) or shape (0,) when dense
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    sm_margin: int,
    negl_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_mask = None
    if negl_prob > 0.0 and tile_stats.dim() == 4:
        is_local = window_size_left >= 0 or (not causal and window_size_right >= 0)
        # FA3 dispatches causal+local as Is_local=True, Is_causal=False.
        # Pure causal (no finite window) stays Is_causal=True, Is_local=False.
        fa3_is_causal = causal and not is_local
        block_mask = _generate_mask_from_stats_mass(
            tile_stats, q.shape[1], q.shape[-1], v.shape[-1], q.dtype, negl_prob, softmax_lse,
            is_causal=fa3_is_causal, is_local=is_local, softcap=softcap,
        )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _flash_attn_backward(
        dout, q, k, v, out, softmax_lse,
        None, None,        # cu_seqlens_q/k
        None, None,        # seqused_q/k
        None, None,        # max_seqlen_q/k
        dq, dk, dv,
        softmax_scale, causal,
        (window_size_left, window_size_right),
        softcap, deterministic, sm_margin,
        block_mask=block_mask,
    )
    dq = dq[..., :q.shape[-1]]
    dk = dk[..., :k.shape[-1]]
    dv = dv[..., :v.shape[-1]]
    return dq, dk, dv


@torch.library.register_fake("back_lite::_flash_attn_bwd")
def _back_lite_bwd_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    tile_stats: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    deterministic: bool,
    sm_margin: int,
    negl_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _back_lite_setup_context(ctx, inputs, output):
    (q, k, v,
     softmax_scale, causal,
     window_size_left, window_size_right,
     softcap, num_splits, deterministic, sm_margin,
     negl_prob, block_n_size, block_m_size) = inputs
    out, softmax_lse, tile_stats = output
    ctx.save_for_backward(q, k, v, out, softmax_lse, tile_stats)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right
    ctx.softcap = softcap
    ctx.deterministic = deterministic
    ctx.sm_margin = sm_margin
    ctx.negl_prob = negl_prob


def _back_lite_backward(ctx, dout, *grads):
    q, k, v, out, softmax_lse, tile_stats = ctx.saved_tensors
    dq, dk, dv = _back_lite_bwd(
        dout, q, k, v, out, softmax_lse, tile_stats,
        ctx.softmax_scale, ctx.causal,
        ctx.window_size_left, ctx.window_size_right,
        ctx.softcap, ctx.deterministic, ctx.sm_margin, ctx.negl_prob,
    )
    # 14 inputs → gradients for q, k, v + None for 11 scalar params
    return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


_back_lite_fwd.register_autograd(
    _back_lite_backward, setup_context=_back_lite_setup_context
)


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        softmax_scale,
        causal,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            None,  # qv
            None,  # out
            None, None, None,   # cu_seqlens_q/k/k_new
            None, None,   # seqused_q/k
            None, None,   # max_seqlen_q/k
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse, *rest)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        ctx.sm_margin = sm_margin
        # return out, softmax_lse
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, *_ = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        if ctx.ndim == 5:
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.unbind(dim=-3)
        else:
            num_heads_q = q.shape[2]
            num_heads_k = k.shape[2]
            qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax_lse=False,
        tile_stats=None,
        negl_prob=0.0,
    ):
        _sparse_enabled = negl_prob > 0
        if _sparse_enabled and tile_stats is None:
            batch, seq_q, heads, head_dim = q.shape
            fwd_is_local = (window_size[0] >= 0 or window_size[1] >= 0) and not causal
            result = flash_attn_3_cuda.get_tile_size_fwd_sm90(
                head_dim, head_dim, causal, fwd_is_local, q.element_size(),
                False, False, False, (q.dtype == torch.int8)
            )
            kBlockM, kBlockN = result[0], result[1]
            if fwd_is_local:
                wl = max(window_size[0], 0)
                wr = max(window_size[1], 0)
                num_n_blocks = (wl + wr + kBlockM + kBlockN - 1) // kBlockN + 1
            else:
                num_n_blocks = (k.shape[1] + kBlockN - 1) // kBlockN
            # Per-row tile LSE stats: logical shape [batch, heads, seq_q, num_n_blocks].
            # Must be -inf so causally-masked positions (never written by fwd kernel)
            # yield exp(-inf)=0 mass instead of garbage from uninitialized memory.
            #
            # Allocate as [B,H,N,T] then permute to [B,H,T,N].
            # This gives stride_m=1 (row dimension is innermost in memory),
            # so the forward kernel writes are coalesced.
            tile_stats = torch.full(
                (batch, heads, num_n_blocks, seq_q), float('-inf'),
                device=q.device, dtype=torch.float32
            ).permute(0, 1, 3, 2)  # shape [B,H,T,N], stride_m=1

        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            None, None, None,   # cu_seqlens_q/k/k_new
            None, None,   # seqused_q/k
            None, None,   # max_seqlen_q/k
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            tile_stats=tile_stats,
        )

        if _sparse_enabled:
            ctx.save_for_backward(q, k, v, out, softmax_lse, tile_stats, *rest)
        else:
            ctx.save_for_backward(q, k, v, out, softmax_lse, *rest)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.negl_prob = negl_prob
        ctx.head_dim_v = v.shape[-1]
        ctx.q_dtype = q.dtype
        if return_softmax_lse:
            return out, softmax_lse
        else:
            return out

    @staticmethod
    def backward(ctx, dout, *args):
        saved = ctx.saved_tensors
        # Unpack saved tensors: q, k, v, out, softmax_lse, [tile_stats or None], [*rest]
        q, k, v, out, softmax_lse = saved[0], saved[1], saved[2], saved[3], saved[4]
        block_mask = None
        negl_prob = getattr(ctx, 'negl_prob', 0.0)
        if negl_prob > 0 and len(saved) > 5:
            tile_stats = saved[5]
            # tile_stats: [B, H, seq_q, num_n_blocks] float32
            if tile_stats is not None and tile_stats.dim() == 4 and tile_stats.dtype == torch.float32:
                is_local = (ctx.window_size[0] >= 0 or ctx.window_size[1] >= 0) and not ctx.causal
                block_mask = _generate_mask_from_stats_mass(
                    tile_stats, q.shape[1], q.shape[-1], ctx.head_dim_v, ctx.q_dtype, negl_prob, softmax_lse,
                    is_causal=ctx.causal, is_local=is_local, softcap=ctx.softcap,
                )

        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None, None, # cu_seqlens_q, cu_seqlens_k,
            None, None, # sequed_q, sequed_k,
            None, None, # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            block_mask=block_mask,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, *((None,) * 16)


class FlashAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *_ = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,   # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        num_heads_q,
        sm_margin,
    )


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_softmax_lse=False,
    tile_stats=None,
    negl_prob=0.0,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    # Fast path: use compile-transparent custom_op when no exotic kwargs are needed.
    # Fall back to legacy FlashAttnFunc (torch.autograd.Function) for rare features.
    _use_fast_path = (
        qv is None
        and q_descale is None and k_descale is None and v_descale is None
        and attention_chunk == 0
        and pack_gqa is None
    )
    if _use_fast_path:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # Determine kBlockM/kBlockN for tile_stats shape (needed only when sparse).
        if negl_prob > 0.0:
            fwd_is_local = (window_size[0] >= 0 or window_size[1] >= 0) and not causal
            tile_info = flash_attn_3_cuda.get_tile_size_fwd_sm90(
                q.shape[-1], v.shape[-1], causal, fwd_is_local,
                q.element_size(), False, False, softcap > 0.0, (q.dtype == torch.int8),
            )
            block_m_size = int(tile_info[0])
            block_n_size = int(tile_info[1])
        else:
            block_m_size = 64  # placeholder; tile_stats unused in dense path
            block_n_size = 64  # placeholder; tile_stats unused in dense path

        out, softmax_lse_out, _ = _back_lite_fwd(
            q, k, v,
            softmax_scale, causal,
            window_size[0], window_size[1],
            softcap, num_splits, deterministic, sm_margin,
            negl_prob, block_n_size, block_m_size,
        )
        if return_softmax_lse:
            return out, softmax_lse_out
        return out

    # Legacy path (rare kwargs: qv, descale, attention_chunk, pack_gqa).
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_softmax_lse,
        tile_stats,
        negl_prob,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale, k_descale, v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
    )


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_3_cuda.fwd_combine(out_partial, lse_partial, out, out_dtype)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            page_block_size can be arbitrary (e.g, 1, 2, 3, 64, etc.).
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v)
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale, k_descale, v_descale,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    # return (out, softmax_lse) if return_softmax_lse else out
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = flash_attn_3_cuda.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0], window_size[1],
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return scheduler_metadata
