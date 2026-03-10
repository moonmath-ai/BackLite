"""
Fused Triton kernel for mask generation from per-block LSE statistics.

tile_lse contains per-block log-sum-exp in log2 domain:
    tile_lse[b, h, row, n] = log2(sum_i exp(score_i))  for KV block n

softmax_lse is the full-row log-sum-exp in natural log (ln) domain:
    softmax_lse[b, h, row] = ln(sum_i exp(score_i))  for all KV blocks

Pipeline:
    [B, H, seq_q, N] tile_lse + [B, H, seq_q] softmax_lse
        → int32[B, H, N, NUM_WORDS] bitmask

Grid = (BH, N): one CTA per (batch-head, KV-block).  Each CTA loops over all
Tm Q-tiles, accumulates tile_mass[Tm] in registers, normalizes + sorts +
thresholds, then packs into NUM_WORDS int32 words — no scratch buffer.
"""

import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# ══════════════════════════════════════════════════════════════════════════
# Fused bitmask kernel — grid=(BH, N), emits int32[BH, N, NUM_WORDS]
#
# For each KV-block n, loops over Tm Q-tiles → computes tile_mass[Tm] in
# registers → normalizes + sorts + thresholds → packs into NUM_WORDS int32 words.
# ══════════════════════════════════════════════════════════════════════════
@triton.jit
def _fused_bitmask_kernel(
    tile_lse_ptr,       # [B, H, seq_q, N] float32, stride_row=1, per-block LSE
    softmax_lse_ptr,    # [B, H, seq_q] float32
    bitmask_ptr,        # [BH, N_ACTUAL, NUM_WORDS] int32 — packed output
    H: tl.constexpr,
    seq_q,
    stride_lse_b,
    stride_lse_h,
    stride_lse_row,     # = 1: consecutive seq_q positions are coalesced
    stride_lse_n,
    stride_slse_b,
    stride_slse_h,
    stride_slse_row,
    negl_prob,
    N_ACTUAL,
    BLOCK_M: tl.constexpr,   # rows per backward m-tile = kBlockM_bwd
    Tm:      tl.constexpr,   # number of Q-block tiles  (≤ TM_PAD)
    TM_PAD:  tl.constexpr,   # max(32, next_power_of_2(Tm)); multiple of 32
    NUM_WORDS: tl.constexpr, # TM_PAD // 32 — int32 words per (bh, n) pair
):
    """Grid = (BH, N_ACTUAL). One CTA → tile_mass[0..Tm-1] for fixed KV-block n → NUM_WORDS int32 words.

    Each output word w packs bits for Q-blocks [w*32 .. (w+1)*32).
    Bit (m % 32) of word (m // 32) is set iff Q-block m is active for this KV-block.
    """
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)   # KV-block column index

    b = pid_bh // H
    h = pid_bh % H

    lse_base  = b * stride_lse_b  + h * stride_lse_h
    slse_base = b * stride_slse_b + h * stride_slse_h

    LOG2E_C: tl.constexpr = 1.4426950408889634

    # ── Compute tile_mass[m] for every Q-tile m, accumulate into [TM_PAD] ──
    tile_mass = tl.zeros([TM_PAD], dtype=tl.float32)

    # Use range() (dynamic loop) to avoid O(Tm) PTX unrolling for large Tm.
    # tl.static_range would produce thousands of unrolled iterations at large
    # Tm, making compilation very slow. range() generates a compact loop.
    for m in range(Tm):
        row_start = m * BLOCK_M
        row_offs  = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        valid_r   = row_offs < seq_q
        safe_rows = tl.minimum(row_offs, seq_q - 1)

        # Load tile_lse[bh, row, n] — stride_row=1, so consecutive rows coalesce
        lse  = tl.load(tile_lse_ptr  + lse_base  + safe_rows * stride_lse_row + pid_n * stride_lse_n,
                       mask=valid_r, other=float('-inf'))
        slse = tl.load(softmax_lse_ptr + slse_base + safe_rows * stride_slse_row,
                       mask=valid_r, other=0.0)

        row_mass = tl.exp2(lse - slse * LOG2E_C)
        row_mass = tl.where(valid_r, row_mass, 0.0)
        mass_m   = tl.sum(row_mass)          # scalar

        # Scatter into the m-th element.  With dynamic m, Triton emits a
        # vectorized compare+select (one warp op over TM_PAD elements) — not a loop.
        tile_mass = tl.where(tl.arange(0, TM_PAD) == m, mass_m, tile_mass)

    # ── Normalize + sort + cumsum + threshold → active[TM_PAD] ─────────────
    valid_m   = tl.arange(0, TM_PAD) < Tm
    total     = tl.sum(tile_mass)
    tile_prob = tile_mass / tl.maximum(total, 1e-20)
    tile_prob = tl.where(valid_m, tile_prob, 0.0)

    sorted_prob = tl.sort(tile_prob)
    csum        = tl.cumsum(sorted_prob, axis=0)

    budget_mask      = csum < negl_prob
    n_to_ignore      = tl.sum(budget_mask.to(tl.int32))
    threshold        = tl.max(tl.where(budget_mask, sorted_prob, float('-inf')))

    n_strictly_below = tl.sum((tile_prob < threshold).to(tl.int32))
    n_ties_to_ignore = n_to_ignore - n_strictly_below
    at_threshold     = (tile_prob == threshold)
    tie_cumcount     = tl.cumsum(at_threshold.to(tl.int32), axis=0)
    ignore_tie       = at_threshold & (tie_cumcount <= n_ties_to_ignore)

    ignore = (tile_prob < threshold) | ignore_tie
    active = tl.where(ignore, 0, 1)
    active = tl.where(total > 1e-12, active, 1)  # keep all when no attention

    # ── Pack active[TM_PAD] → NUM_WORDS int32 bitmask words ─────────────────
    # Reshape [TM_PAD] = [NUM_WORDS * 32] → [NUM_WORDS, 32]:
    #   row w = bits for Q-blocks [w*32 .. (w+1)*32).
    # valid_m_2d ensures padding bits (m >= Tm) are always 0.
    active_2d   = tl.reshape(active,               [NUM_WORDS, 32])
    valid_m_2d  = tl.reshape(valid_m.to(tl.int32), [NUM_WORDS, 32])
    bit_offs_2d = tl.broadcast_to(tl.arange(0, 32)[None, :], [NUM_WORDS, 32])
    bits_2d     = tl.where((valid_m_2d != 0) & (active_2d != 0),
                            1 << bit_offs_2d, 0).to(tl.int32)
    bitmask_words = tl.sum(bits_2d, axis=1)   # [NUM_WORDS]

    w_offs = tl.arange(0, NUM_WORDS)
    tl.store(bitmask_ptr + (pid_bh * N_ACTUAL + pid_n) * NUM_WORDS + w_offs, bitmask_words)


# Max Tm supported by the bitmask kernel.
# Bounded by tl.sort's limit (2048 elements) — TM_PAD must be a power-of-2 ≤ 2048.
# 2048 → T ≤ 128K at kBlockM=64, T ≤ 256K at kBlockM=128.
_BITMASK_MAX_TM = 2048


def mask_from_stats_fused(
    tile_lse: torch.Tensor,      # [B, H, seq_q, N] float32 — per-row cumulative LSE (log2 domain)
    softmax_lse: torch.Tensor,   # [B, H, seq_q] float32 — full-row LSE (ln domain)
    num_row_tiles_bwd: int,      # Tm = number of backward m-tiles
    kBlockM_bwd: int,            # rows per backward tile
    negl_prob: float,
) -> torch.Tensor:
    """
    Fused mask generation: per-block LSE → packed int32 bitmask.

    Uses _fused_bitmask_kernel with grid=(BH, N).  Each CTA loops over all Tm
    Q-tiles for one KV-column, accumulates tile_mass[Tm] in registers,
    thresholds, and packs directly into NUM_WORDS int32 words — no scratch
    buffer, no intermediate uint8.

    Bit (m % 32) of word (m // 32) of bitmask[b, h, n, :] is set iff Q-block m
    should be computed for KV-block n.  The C++ backward kernel streams words
    on-demand via __ldg and uses __ffs() for O(1) next-active-m lookups.

    Returns:
        int32 bitmask of shape [B, H, N, NUM_WORDS]
        where NUM_WORDS = max(32, next_power_of_2(Tm)) // 32

    Raises:
        AssertionError if Tm > 2048 (tl.sort limit) or N > 256K (kBlockN=128).
    """
    B, H, seq_q, N = tile_lse.shape
    BH = B * H
    Tm = num_row_tiles_bwd

    assert tile_lse.dtype == torch.float32, f"tile_lse must be float32, got {tile_lse.dtype}"
    assert softmax_lse.dtype == torch.float32, f"softmax_lse must be float32, got {softmax_lse.dtype}"
    assert softmax_lse.shape[:2] == (B, H), \
        f"softmax_lse batch/head {softmax_lse.shape[:2]} doesn't match tile_lse (B={B}, H={H})"
    assert Tm <= _BITMASK_MAX_TM, \
        f"Tm={Tm} exceeds max supported ({_BITMASK_MAX_TM}); tl.sort limit is 2048 elements"

    # N_PAD must be power of 2 for tl.sort; max 2048 → seq_len ≤ 256K at kBlockN=128
    N_PAD = _next_power_of_2(N)
    assert N_PAD <= 2048, f"N_PAD={N_PAD} too large for Triton sort (max 2048)"

    stride_lse_b, stride_lse_h, stride_lse_row, stride_lse_n = tile_lse.stride()
    stride_slse_b, stride_slse_h, stride_slse_row = softmax_lse.stride()

    # TM_PAD must be a multiple of 32 (word size) and a power-of-2 for tl.sort.
    TM_PAD = max(32, _next_power_of_2(Tm))
    NUM_WORDS = TM_PAD // 32
    bitmask_out = torch.empty(BH, N, NUM_WORDS, device=tile_lse.device, dtype=torch.int32)

    # num_warps: scale with TM_PAD (loop iteration count determines latency)
    if TM_PAD <= 32:
        num_warps = 1
    elif TM_PAD <= 128:
        num_warps = 2
    elif TM_PAD <= 512:
        num_warps = 4
    else:
        num_warps = 8

    _fused_bitmask_kernel[(BH, N)](
        tile_lse, softmax_lse, bitmask_out,
        H=H,
        seq_q=seq_q,
        stride_lse_b=stride_lse_b,
        stride_lse_h=stride_lse_h,
        stride_lse_row=stride_lse_row,
        stride_lse_n=stride_lse_n,
        stride_slse_b=stride_slse_b,
        stride_slse_h=stride_slse_h,
        stride_slse_row=stride_slse_row,
        negl_prob=negl_prob,
        N_ACTUAL=N,
        BLOCK_M=kBlockM_bwd,
        Tm=Tm,
        TM_PAD=TM_PAD,
        NUM_WORDS=NUM_WORDS,
        num_warps=num_warps,
    )
    return bitmask_out.view(B, H, N, NUM_WORDS)
