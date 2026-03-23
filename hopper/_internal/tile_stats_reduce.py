"""
Mask generation from per-block LSE statistics.

Output bitmask layout: [B, H, N, NUM_WORDS] int32
  bit (m % 32) of word (m // 32) at column n is set iff the (m, n) tile
  is active.

Strategy
--------
  1.  Compute tile_mass[BH, Tm, N] — unnormalised mass per (row-tile, col-tile).
  2.  For each (bh, m): normalise across N, apply smallest_mass_subset
      to find which columns to prune, and scatter the per-row decision
      into the column-major bitmask.

  Tm ≤ 128 (N ≤ sort-limit):  fused single kernel, grid = (BH, Tm)
  Tm > 128:                     two-kernel path (mass + threshold/pack)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Fused kernel (N ≤ sort-limit): grid = (BH, Tm)
# Each program handles one (batch*head, row-tile) and produces one bit per column.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _fused_rowwise_bitmask_kernel(
    tile_lse_ptr,       # [B, H, seq_q, N] float32, stride_row=1
    softmax_lse_ptr,    # [B, H, seq_q]    float32
    bitmask_ptr,        # [BH, N, NUM_WORDS] int32 — output (atomically OR'd)
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
    N_ACTUAL,           # number of K-column tiles
    BLOCK_M:  tl.constexpr,
    Tm,                 # number of Q-row-tiles (runtime)
    N_PAD:    tl.constexpr,  # next-power-of-2 of N_ACTUAL
    NUM_WORDS: tl.constexpr, # Tm packed into int32 words
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    if pid_m >= Tm:
        return

    b = pid_bh // H
    h = pid_bh % H
    lse_base  = b.to(tl.int64) * stride_lse_b  + h.to(tl.int64) * stride_lse_h
    slse_base = b.to(tl.int64) * stride_slse_b + h.to(tl.int64) * stride_slse_h
    LOG2E_C: tl.constexpr = 1.4426950408889634

    # ── Compute tile_mass[n] for this row-tile m ────────────────────────
    tile_mass = tl.zeros([N_PAD], dtype=tl.float32)
    row_start = pid_m * BLOCK_M
    for r in range(BLOCK_M):
        row = row_start + r
        valid_r = row < seq_q
        safe_row = tl.minimum(row, seq_q - 1)
        slse_val = tl.load(softmax_lse_ptr + slse_base + safe_row * stride_slse_row)
        n_offs = tl.arange(0, N_PAD)
        valid_n = n_offs < N_ACTUAL
        lse_vals = tl.load(
            tile_lse_ptr + lse_base + safe_row * stride_lse_row + n_offs * stride_lse_n,
            mask=valid_n & valid_r, other=float('-inf'))
        row_mass = tl.exp2(lse_vals - slse_val * LOG2E_C)
        row_mass = tl.where(valid_n & valid_r, row_mass, 0.0)
        tile_mass += row_mass

    # ── Normalise across columns (dim=-1) ───────────────────────────────
    valid_n = tl.arange(0, N_PAD) < N_ACTUAL
    total = tl.sum(tile_mass)
    tile_prob = tl.where(valid_n, tile_mass / tl.maximum(total, 1e-20), 0.0)

    # ── smallest_mass_subset along columns ──────────────────────────────
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
    active = tl.where(valid_n, active, 0)

    # ── Scatter results into column-major bitmask ───────────────────────
    # bitmask layout: [BH, N, NUM_WORDS] — bit (m%32) of word (m//32) at column n
    m = pid_m
    word_idx = m >> 5       # m // 32
    bit_val  = 1 << (m & 31)  # 1 << (m % 32)

    # For each active column n, atomically OR the bit into bitmask[bh, n, word_idx]
    n_offs = tl.arange(0, N_PAD)
    out_ptrs = bitmask_ptr + (pid_bh.to(tl.int64) * N_ACTUAL + n_offs) * NUM_WORDS + word_idx
    # Use atomic_or so that multiple row-tile programs can write concurrently
    tl.atomic_or(out_ptrs, tl.where(active != 0, bit_val, 0).to(tl.int32), mask=n_offs < N_ACTUAL)


# ═══════════════════════════════════════════════════════════════════════════════
# Two-kernel path (Tm > sort-limit or N > sort-limit)
# ═══════════════════════════════════════════════════════════════════════════════

# Kernel 1: compute tile_mass[BH, Tm, N], grid = (BH, ceil(Tm/CHUNK), N_GROUPS)
@triton.jit
def _tile_mass_rowwise_kernel(
    tile_lse_ptr,       # [B, H, seq_q, N] float32
    softmax_lse_ptr,    # [B, H, seq_q]    float32
    tile_mass_ptr,      # [BH, Tm, N]      float32  — output
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
    N_BLOCK: tl.constexpr,
):
    pid_bh    = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_ng    = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    lse_base  = b.to(tl.int64) * stride_lse_b  + h.to(tl.int64) * stride_lse_h
    slse_base = b.to(tl.int64) * stride_slse_b + h.to(tl.int64) * stride_slse_h

    LOG2E_C: tl.constexpr = 1.4426950408889634

    n_start = pid_ng * N_BLOCK
    n_offs  = n_start + tl.arange(0, N_BLOCK)
    valid_n = n_offs < N_ACTUAL

    m_base = pid_chunk * CHUNK
    for i in tl.static_range(CHUNK):
        m       = m_base + i
        valid_m = m < Tm
        row_start = m * BLOCK_M

        mass_acc = tl.zeros([N_BLOCK], dtype=tl.float32)
        for r in range(BLOCK_M):
            row = row_start + r
            valid_r = valid_m & (row < seq_q)
            safe_row = tl.minimum(row, seq_q - 1) if valid_m else 0
            slse_val = tl.load(softmax_lse_ptr + slse_base + safe_row * stride_slse_row)
            lse_vals = tl.load(
                tile_lse_ptr + lse_base + safe_row * stride_lse_row + n_offs * stride_lse_n,
                mask=valid_n & valid_r, other=float('-inf'))
            row_mass = tl.exp2(lse_vals - slse_val * LOG2E_C)
            row_mass = tl.where(valid_n & valid_r, row_mass, 0.0)
            mass_acc += row_mass

        # Store tile_mass[bh, m, n_start:n_start+N_BLOCK]
        out_base = (pid_bh.to(tl.int64) * Tm + m) * N_ACTUAL + n_offs
        tl.store(tile_mass_ptr + out_base, mass_acc, mask=valid_m & valid_n)


# Kernel 2: normalise per-row, threshold, scatter into bitmask.  grid = (BH, Tm)
@triton.jit
def _threshold_scatter_kernel(
    tile_mass_ptr,  # [BH, Tm, N] float32
    bitmask_ptr,    # [BH, N, NUM_WORDS] int32  — atomically OR'd
    N_ACTUAL,
    Tm,
    negl_prob,
    N_PAD:     tl.constexpr,
    NUM_WORDS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    if pid_m >= Tm:
        return

    base = (pid_bh.to(tl.int64) * Tm + pid_m) * N_ACTUAL
    n_offs  = tl.arange(0, N_PAD)
    valid_n = n_offs < N_ACTUAL
    tile_mass = tl.load(tile_mass_ptr + base + n_offs, mask=valid_n, other=0.0)

    total     = tl.sum(tile_mass)
    tile_prob = tl.where(valid_n, tile_mass / tl.maximum(total, 1e-20), 0.0)

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
    active = tl.where(valid_n, active, 0)

    # Scatter: bitmask[bh, n, word_idx] |= bit_val  for each active n
    m = pid_m
    word_idx = m >> 5
    bit_val  = 1 << (m & 31)
    out_ptrs = bitmask_ptr + (pid_bh.to(tl.int64) * N_ACTUAL + n_offs) * NUM_WORDS + word_idx
    tl.atomic_or(out_ptrs, tl.where(active != 0, bit_val, 0).to(tl.int32), mask=valid_n)


# Max N supported by tl.sort: power-of-2 ≤ 2048.
_SORT_LIMIT = 2048

# N threshold below which fused single-kernel is faster (measured on H100).
_FUSED_N_THRESHOLD = 128


def mask_from_stats_fused(
    tile_lse: torch.Tensor,    # [B, H, seq_q, N] float32
    softmax_lse: torch.Tensor, # [B, H, seq_q]    float32
    num_row_tiles_bwd: int,    # Tm
    kBlockM_bwd: int,          # BLOCK_M
    negl_prob: float,
) -> torch.Tensor:
    """
    Row-wise mask generation: per-block LSE → packed int32 bitmask.

    For each Q-row-tile m, normalises probability mass across K-column-tiles,
    then prunes the smallest columns whose combined mass ≤ negl_prob.

    Returns int32[B, H, N, NUM_WORDS].
    """
    B, H, seq_q, N = tile_lse.shape
    BH = B * H
    Tm = num_row_tiles_bwd

    assert tile_lse.dtype    == torch.float32
    assert softmax_lse.dtype == torch.float32
    assert softmax_lse.shape[:2] == (B, H)

    stride_lse_b, stride_lse_h, stride_lse_row, stride_lse_n = tile_lse.stride()
    stride_slse_b, stride_slse_h, stride_slse_row = softmax_lse.stride()

    TM_PAD    = max(32, _next_power_of_2(Tm))
    NUM_WORDS = TM_PAD // 32
    N_PAD     = max(32, _next_power_of_2(N))

    # Output bitmask: zero-initialised (atomic_or requires clean start)
    bitmask_out = torch.zeros(BH, N, NUM_WORDS, device=tile_lse.device, dtype=torch.int32)

    if N <= _FUSED_N_THRESHOLD and N_PAD <= _SORT_LIMIT:
        # ── Fused single kernel: grid = (BH, Tm) ───────────────────────
        num_warps = 2 if N_PAD <= 64 else 4

        _fused_rowwise_bitmask_kernel[(BH, Tm)](
            tile_lse, softmax_lse, bitmask_out,
            H=H, seq_q=seq_q,
            stride_lse_b=stride_lse_b, stride_lse_h=stride_lse_h,
            stride_lse_row=stride_lse_row, stride_lse_n=stride_lse_n,
            stride_slse_b=stride_slse_b, stride_slse_h=stride_slse_h,
            stride_slse_row=stride_slse_row,
            negl_prob=negl_prob, N_ACTUAL=N,
            BLOCK_M=kBlockM_bwd, Tm=Tm, N_PAD=N_PAD, NUM_WORDS=NUM_WORDS,
            num_warps=num_warps,
        )
    else:
        # ── Two-kernel path ─────────────────────────────────────────────
        assert N_PAD <= _SORT_LIMIT, f"N={N} exceeds sort limit {_SORT_LIMIT}"

        # Kernel 1: compute tile_mass[BH, Tm, N]
        CHUNK     = 16
        N_BLOCK   = min(N_PAD, 128)
        n_groups  = (N + N_BLOCK - 1) // N_BLOCK
        num_chunks = (Tm + CHUNK - 1) // CHUNK
        tile_mass  = torch.zeros(BH, Tm, N, device=tile_lse.device, dtype=torch.float32)

        _tile_mass_rowwise_kernel[(BH, num_chunks, n_groups)](
            tile_lse, softmax_lse, tile_mass,
            H=H, seq_q=seq_q,
            stride_lse_b=stride_lse_b, stride_lse_h=stride_lse_h,
            stride_lse_row=stride_lse_row, stride_lse_n=stride_lse_n,
            stride_slse_b=stride_slse_b, stride_slse_h=stride_slse_h,
            stride_slse_row=stride_slse_row,
            N_ACTUAL=N, Tm=Tm, BLOCK_M=kBlockM_bwd,
            CHUNK=CHUNK, N_BLOCK=N_BLOCK,
            num_warps=2,
        )

        # Kernel 2: normalise per-row, threshold, scatter bits
        num_warps2 = 2 if N_PAD <= 64 else (4 if N_PAD <= 256 else 8)

        _threshold_scatter_kernel[(BH, Tm)](
            tile_mass, bitmask_out,
            N_ACTUAL=N, Tm=Tm, negl_prob=negl_prob,
            N_PAD=N_PAD, NUM_WORDS=NUM_WORDS,
            num_warps=num_warps2,
        )

    return bitmask_out.view(B, H, N, NUM_WORDS)
