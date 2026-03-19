"""
Mask generation from per-block LSE statistics.

  Tm ≤ 128  →  fused single kernel (grid=BH×N, no scratch buffer)
  Tm > 128  →  two-kernel path (parallel tile_mass + sort/pack)
  Crossover at ~Tm=128 (seq≈8k at kBlockM=64) on H100.

tile_lse:    [B, H, seq_q, N]  float32
softmax_lse: [B, H, seq_q]     float32
→ bitmask:   [B, H, N, NUM_WORDS]  int32
"""

import torch
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# Fused kernel (Tm ≤ 128): grid=(BH, N), no scratch buffer.
@triton.jit
def _fused_bitmask_kernel(
    tile_lse_ptr,       # [B, H, seq_q, N] float32, stride_row=1
    softmax_lse_ptr,    # [B, H, seq_q]    float32
    bitmask_ptr,        # [BH, N, NUM_WORDS] int32 — output
    H: tl.constexpr,
    seq_q,
    stride_lse_b,
    stride_lse_h,
    stride_lse_row,
    stride_lse_n,
    stride_slse_b,
    stride_slse_h,
    stride_slse_row,
    negl_prob,
    N_ACTUAL,
    BLOCK_M:   tl.constexpr,
    Tm:        tl.constexpr,
    TM_PAD:    tl.constexpr,
    NUM_WORDS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    lse_base  = b.to(tl.int64) * stride_lse_b  + h.to(tl.int64) * stride_lse_h
    slse_base = b.to(tl.int64) * stride_slse_b + h.to(tl.int64) * stride_slse_h
    LOG2E_C: tl.constexpr = 1.4426950408889634

    tile_mass = tl.zeros([TM_PAD], dtype=tl.float32)
    for m in range(Tm):
        row_offs  = m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_r   = row_offs < seq_q
        safe_rows = tl.minimum(row_offs, seq_q - 1)
        lse  = tl.load(tile_lse_ptr    + lse_base  + safe_rows * stride_lse_row + pid_n * stride_lse_n,
                       mask=valid_r, other=float('-inf'))
        slse = tl.load(softmax_lse_ptr + slse_base + safe_rows * stride_slse_row,
                       mask=valid_r, other=0.0)
        row_mass  = tl.where(valid_r, tl.exp2(lse - slse * LOG2E_C), 0.0)
        tile_mass = tl.where(tl.arange(0, TM_PAD) == m, tl.sum(row_mass), tile_mass)

    valid_m   = tl.arange(0, TM_PAD) < Tm
    total     = tl.sum(tile_mass)
    tile_prob = tl.where(valid_m, tile_mass / tl.maximum(total, 1e-20), 0.0)

    sorted_prob      = tl.sort(tile_prob)
    csum             = tl.cumsum(sorted_prob, axis=0)
    budget_mask      = csum < negl_prob
    n_to_ignore      = tl.sum(budget_mask.to(tl.int32))
    threshold        = tl.max(tl.where(budget_mask, sorted_prob, float('-inf')))
    n_strictly_below = tl.sum((tile_prob < threshold).to(tl.int32))
    n_ties_to_ignore = n_to_ignore - n_strictly_below
    at_threshold     = tile_prob == threshold
    ignore_tie       = at_threshold & (tl.cumsum(at_threshold.to(tl.int32), axis=0) <= n_ties_to_ignore)
    active = tl.where((tile_prob < threshold) | ignore_tie, 0, 1)
    active = tl.where(total > 1e-12, active, 1)

    active_2d   = tl.reshape(active,               [NUM_WORDS, 32])
    valid_m_2d  = tl.reshape(valid_m.to(tl.int32), [NUM_WORDS, 32])
    bit_offs_2d = tl.broadcast_to(tl.arange(0, 32)[None, :], [NUM_WORDS, 32])
    bits_2d     = tl.where((valid_m_2d != 0) & (active_2d != 0), 1 << bit_offs_2d, 0).to(tl.int32)
    tl.store(bitmask_ptr + (pid_bh.to(tl.int64) * N_ACTUAL + pid_n) * NUM_WORDS
             + tl.arange(0, NUM_WORDS), tl.sum(bits_2d, axis=1))


# Kernel 1 (Tm > 128): compute tile_mass[BH, N, Tm], grid=(BH, N, ceil(Tm/CHUNK)).
@triton.jit
def _tile_mass_kernel(
    tile_lse_ptr,       # [B, H, seq_q, N] float32, stride_row=1
    softmax_lse_ptr,    # [B, H, seq_q]    float32
    tile_mass_ptr,      # [BH, N, Tm]      float32  — output
    H: tl.constexpr,
    seq_q,
    stride_lse_b,
    stride_lse_h,
    stride_lse_row,
    stride_lse_n,
    stride_slse_b,
    stride_slse_h,
    stride_slse_row,
    N_ACTUAL,
    Tm,
    BLOCK_M: tl.constexpr,
    CHUNK:   tl.constexpr,
):
    pid_bh    = tl.program_id(0)
    pid_n     = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    lse_base  = b.to(tl.int64) * stride_lse_b  + h.to(tl.int64) * stride_lse_h
    slse_base = b.to(tl.int64) * stride_slse_b + h.to(tl.int64) * stride_slse_h
    n_off     = pid_n.to(tl.int64) * stride_lse_n

    LOG2E_C: tl.constexpr = 1.4426950408889634

    m_base   = pid_chunk * CHUNK
    out_base = (pid_bh.to(tl.int64) * N_ACTUAL + pid_n.to(tl.int64)) * Tm + m_base

    for i in tl.static_range(CHUNK):
        m         = m_base + i
        valid_m   = m < Tm
        row_start = m * BLOCK_M
        row_offs  = row_start + tl.arange(0, BLOCK_M)
        valid_r   = valid_m & (row_offs < seq_q)
        safe_rows = tl.minimum(row_offs, seq_q - 1)

        lse  = tl.load(tile_lse_ptr    + lse_base  + safe_rows * stride_lse_row + n_off,
                       mask=valid_r, other=float('-inf'))
        slse = tl.load(softmax_lse_ptr + slse_base + safe_rows * stride_slse_row,
                       mask=valid_r, other=0.0)

        row_mass = tl.exp2(lse - slse * LOG2E_C)
        row_mass = tl.where(valid_r, row_mass, 0.0)
        mass     = tl.sum(row_mass)

        tl.store(tile_mass_ptr + out_base + i, mass, mask=valid_m)


# Kernel 2 (Tm > 128): normalise + sort + threshold + pack, grid=(BH, N).
@triton.jit
def _threshold_pack_kernel(
    tile_mass_ptr,  # [BH, N, Tm]      float32
    bitmask_ptr,    # [BH, N, NUM_WORDS] int32
    N_ACTUAL,
    Tm,
    negl_prob,
    TM_PAD:    tl.constexpr,
    NUM_WORDS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)

    base = (pid_bh.to(tl.int64) * N_ACTUAL + pid_n.to(tl.int64)) * Tm

    m_offs    = tl.arange(0, TM_PAD)
    valid_m   = m_offs < Tm
    tile_mass = tl.load(tile_mass_ptr + base + m_offs, mask=valid_m, other=0.0)

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
    at_threshold     = tile_prob == threshold
    ignore_tie       = at_threshold & (tl.cumsum(at_threshold.to(tl.int32), axis=0) <= n_ties_to_ignore)
    active = tl.where((tile_prob < threshold) | ignore_tie, 0, 1)
    active = tl.where(total > 1e-12, active, 1)

    active_2d   = tl.reshape(active,               [NUM_WORDS, 32])
    valid_m_2d  = tl.reshape(valid_m.to(tl.int32), [NUM_WORDS, 32])
    bit_offs_2d = tl.broadcast_to(tl.arange(0, 32)[None, :], [NUM_WORDS, 32])
    bits_2d     = tl.where((valid_m_2d != 0) & (active_2d != 0), 1 << bit_offs_2d, 0).to(tl.int32)
    tl.store(bitmask_ptr + (pid_bh.to(tl.int64) * N_ACTUAL + pid_n) * NUM_WORDS
             + tl.arange(0, NUM_WORDS), tl.sum(bits_2d, axis=1))


# Max Tm supported (tl.sort limit: power-of-2 ≤ 2048).
_BITMASK_MAX_TM = 2048

# Tm threshold below which the fused single-kernel is faster (measured on H100).
_FUSED_TM_THRESHOLD = 128


def mask_from_stats_fused(
    tile_lse: torch.Tensor,    # [B, H, seq_q, N] float32
    softmax_lse: torch.Tensor, # [B, H, seq_q]    float32
    num_row_tiles_bwd: int,    # Tm
    kBlockM_bwd: int,          # BLOCK_M
    negl_prob: float,
) -> torch.Tensor:
    """
    Mask generation: per-block LSE → packed int32 bitmask.

    Dispatches to fused single-kernel (Tm ≤ 128) or two-kernel path (Tm > 128).
    Returns int32[B, H, N, NUM_WORDS].
    """
    B, H, seq_q, N = tile_lse.shape
    BH = B * H
    Tm = num_row_tiles_bwd

    assert tile_lse.dtype    == torch.float32
    assert softmax_lse.dtype == torch.float32
    assert softmax_lse.shape[:2] == (B, H)
    assert Tm <= _BITMASK_MAX_TM, f"Tm={Tm} > {_BITMASK_MAX_TM} (tl.sort limit)"

    stride_lse_b, stride_lse_h, stride_lse_row, stride_lse_n = tile_lse.stride()
    stride_slse_b, stride_slse_h, stride_slse_row = softmax_lse.stride()

    TM_PAD    = max(32, _next_power_of_2(Tm))
    NUM_WORDS = TM_PAD // 32

    bitmask_out = torch.empty(BH, N, NUM_WORDS, device=tile_lse.device, dtype=torch.int32)

    if Tm <= _FUSED_TM_THRESHOLD:
        if TM_PAD <= 32:   num_warps = 1
        elif TM_PAD <= 128: num_warps = 2
        else:               num_warps = 4

        _fused_bitmask_kernel[(BH, N)](
            tile_lse, softmax_lse, bitmask_out,
            H=H, seq_q=seq_q,
            stride_lse_b=stride_lse_b, stride_lse_h=stride_lse_h,
            stride_lse_row=stride_lse_row, stride_lse_n=stride_lse_n,
            stride_slse_b=stride_slse_b, stride_slse_h=stride_slse_h,
            stride_slse_row=stride_slse_row,
            negl_prob=negl_prob, N_ACTUAL=N,
            BLOCK_M=kBlockM_bwd, Tm=Tm, TM_PAD=TM_PAD, NUM_WORDS=NUM_WORDS,
            num_warps=num_warps,
        )
    else:
        CHUNK      = 16
        num_chunks = (Tm + CHUNK - 1) // CHUNK
        tile_mass  = torch.empty(BH, N, Tm, device=tile_lse.device, dtype=torch.float32)

        _tile_mass_kernel[(BH, N, num_chunks)](
            tile_lse, softmax_lse, tile_mass,
            H=H, seq_q=seq_q,
            stride_lse_b=stride_lse_b, stride_lse_h=stride_lse_h,
            stride_lse_row=stride_lse_row, stride_lse_n=stride_lse_n,
            stride_slse_b=stride_slse_b, stride_slse_h=stride_slse_h,
            stride_slse_row=stride_slse_row,
            N_ACTUAL=N, Tm=Tm, BLOCK_M=kBlockM_bwd, CHUNK=CHUNK,
            num_warps=2,
        )

        if TM_PAD <= 256:   num_warps2 = 2
        elif TM_PAD <= 1024: num_warps2 = 4
        else:                num_warps2 = 8

        _threshold_pack_kernel[(BH, N)](
            tile_mass, bitmask_out,
            N_ACTUAL=N, Tm=Tm, negl_prob=negl_prob,
            TM_PAD=TM_PAD, NUM_WORDS=NUM_WORDS,
            num_warps=num_warps2,
        )

    return bitmask_out.view(B, H, N, NUM_WORDS)
