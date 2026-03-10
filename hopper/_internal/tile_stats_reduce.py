"""
Fused Triton kernel for mask generation from per-block LSE statistics.

tile_lse contains per-block log-sum-exp in log2 domain:
    tile_lse[b, h, row, n] = log2(sum_i exp(score_i))  for KV block n

softmax_lse is the full-row log-sum-exp in natural log (ln) domain:
    softmax_lse[b, h, row] = ln(sum_i exp(score_i))  for all KV blocks

Pipeline:
    [B, H, seq_q, N] tile_lse + [B, H, seq_q] softmax_lse → [B, H, Tm, N] uint8 mask

The forward kernel writes tile_lse with stride_row=1 (permute trick for coalesced
forward writes). This kernel exploits that layout by vectorizing over rows (coalesced)
and looping over n-blocks, avoiding an expensive transpose copy.
"""

import math
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
# Kernel 1: Reduce BLOCK_M rows → tile_mass[N] per (batch-head, tile)
#
# Processes [BLOCK_M, BLOCK_N] 2D tile
# Writes tile_mass vectors to a scratch buffer for kernel 2.
# ══════════════════════════════════════════════════════════════════════════
@triton.jit
def _reduce_tile_mass_kernel(
    tile_lse_ptr,       # [B, H, seq_q, N] float32, stride_row=1, PER-BLOCK LSE
    softmax_lse_ptr,    # [B, H, seq_q] float32
    scratch_ptr,        # [BH, Tm, N_PAD] float32 — tile_mass output
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_PAD: tl.constexpr,
    Tm: tl.constexpr,
):
    """Grid = (B*H, Tm).  Each program reduces BLOCK_M rows × N blocks → tile_mass[N].

    For per-block LSE data, the mass for block n at row r is simply:
        mass(r, n) = exp2(tile_lse[r, n] - softmax_lse[r] * log2(e))
    which is the fraction of total attention probability in KV block n.
    """
    pid_bh   = tl.program_id(0)
    pid_tile = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    row_offs  = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    row_start = pid_tile * BLOCK_M
    rows      = row_start + row_offs
    valid_rows = rows < seq_q

    # ── Base pointers ─────────────────────────────────────────
    lse_base  = b * stride_lse_b  + h * stride_lse_h
    slse_base = b * stride_slse_b + h * stride_slse_h
    scratch_base = pid_bh * Tm * N_PAD + pid_tile * N_PAD

    LOG2E_C: tl.constexpr = 1.4426950408889634

    # Pre-load softmax_lse for all rows
    safe_rows = tl.minimum(rows, seq_q - 1)
    slse_log2 = tl.load(
        softmax_lse_ptr + slse_base + safe_rows * stride_slse_row
    ) * LOG2E_C  # [BLOCK_M], pre-converted to log2

    # ── N-major loop (2D vectorized: BLOCK_N n-blocks per iter) ──
    n_base_offs = tl.arange(0, BLOCK_N)  # [BLOCK_N]

    for n_start in range(0, N_PAD, BLOCK_N):
        n_offs = n_start + n_base_offs            # [BLOCK_N]
        valid_n = n_offs < N_ACTUAL

        # Load [BLOCK_M, BLOCK_N] tile — rows are coalesced (stride_row=1)
        ptrs = (tile_lse_ptr + lse_base
                + rows[:, None] * stride_lse_row
                + n_offs[None, :] * stride_lse_n)
        lse = tl.load(ptrs,
                       mask=valid_rows[:, None] & valid_n[None, :],
                       other=float('-inf'))

        # Per-row probability mass for BLOCK_N blocks:
        #   mass(r, n) = exp2(per_block_lse[r, n] - total_lse[r] * log2e)
        row_mass = tl.exp2(lse - slse_log2[:, None])
        row_mass = tl.where(valid_rows[:, None], row_mass, 0.0)

        # Reduce across BLOCK_M rows → [BLOCK_N] tile_mass values
        mass = tl.sum(row_mass, axis=0)
        tl.store(scratch_ptr + scratch_base + n_offs, mass, mask=valid_n)


# ══════════════════════════════════════════════════════════════════════════
# Kernel 2: Threshold tile_mass → uint8 mask
#
# Loads tile_mass[N] from scratch, normalizes, sorts, cumsum, thresholds.
# Much smaller and faster than kernel 1 (compute-only, ~N values).
# ══════════════════════════════════════════════════════════════════════════
@triton.jit
def _threshold_mask_kernel(
    scratch_ptr,        # [BH, Tm, N_PAD] float32 — tile_mass input
    mask_out_ptr,       # [BH, Tm, N_ACTUAL] uint8 — output mask
    negl_prob,
    N_ACTUAL,
    N_PAD: tl.constexpr,
    Tm: tl.constexpr,
):
    """Grid = (BH, Tm).  Each program thresholds one tile_mass vector → mask."""
    pid_bh   = tl.program_id(0)
    pid_tile = tl.program_id(1)

    n_offs = tl.arange(0, N_PAD)
    valid_n = n_offs < N_ACTUAL
    scratch_base = pid_bh * Tm * N_PAD + pid_tile * N_PAD

    # Load tile_mass from scratch
    tile_mass = tl.load(scratch_ptr + scratch_base + n_offs, mask=valid_n, other=0.0)

    # ── Normalize to probability distribution ─────────────────
    total = tl.sum(tile_mass)
    tile_prob = tile_mass / tl.maximum(total, 1e-20)
    tile_prob = tl.where(valid_n, tile_prob, 0.0)

    # ── Sort + cumsum + threshold → mask ──────────────────────
    sorted_prob = tl.sort(tile_prob)
    csum = tl.cumsum(sorted_prob, axis=0)

    # Tiles whose cumulative ascending probability < negl_prob are negligible
    budget_mask  = csum < negl_prob
    n_to_ignore  = tl.sum(budget_mask.to(tl.int32))

    # Threshold: largest probability within the ignore budget
    threshold = tl.max(tl.where(budget_mask, sorted_prob, float('-inf')))

    # Tie-breaking: among tiles with prob == threshold, use position order
    n_strictly_below = tl.sum((tile_prob < threshold).to(tl.int32))
    n_ties_to_ignore = n_to_ignore - n_strictly_below
    at_threshold     = (tile_prob == threshold)
    tie_cumcount     = tl.cumsum(at_threshold.to(tl.int32), axis=0)
    ignore_tie       = at_threshold & (tie_cumcount <= n_ties_to_ignore)

    ignore = (tile_prob < threshold) | ignore_tie
    mask   = tl.where(ignore, 0, 1).to(tl.uint8)

    # Fallback: if no valid attention at all, keep every tile
    mask = tl.where(total > 1e-12, mask, 1).to(tl.uint8)

    # ── Write output ──────────────────────────────────────────
    out_offset = pid_bh * Tm * N_ACTUAL + pid_tile * N_ACTUAL + n_offs
    tl.store(mask_out_ptr + out_offset, mask, mask=valid_n)


# ══════════════════════════════════════════════════════════════════════════
# Kernel 3: Fused single-kernel path (for N_PAD ≤ 512)
#
# Combines both kernels: reduce BLOCK_M rows → tile_mass[N_PAD], then
# immediately normalize + sort + cumsum + threshold → uint8 mask[N_ACTUAL].
# No scratch buffer needed — everything stays in registers.
# ══════════════════════════════════════════════════════════════════════════
@triton.jit
def _fused_mask_kernel(
    tile_lse_ptr,       # [B, H, seq_q, N] float32, stride_row=1, PER-BLOCK LSE
    softmax_lse_ptr,    # [B, H, seq_q] float32
    mask_out_ptr,       # [BH, Tm, N_ACTUAL] uint8 — output mask
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
    N_PAD: tl.constexpr,   # power-of-2 ≥ N_ACTUAL, ≤ 512
    BLOCK_M: tl.constexpr,
    Tm: tl.constexpr,
):
    """Grid = (B*H, Tm).  Fused reduce + threshold in a single kernel, no scratch."""
    pid_bh   = tl.program_id(0)
    pid_tile = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    row_offs  = tl.arange(0, BLOCK_M)   # [BLOCK_M]
    row_start = pid_tile * BLOCK_M
    rows      = row_start + row_offs
    valid_rows = rows < seq_q

    lse_base  = b * stride_lse_b  + h * stride_lse_h
    slse_base = b * stride_slse_b + h * stride_slse_h

    LOG2E_C: tl.constexpr = 1.4426950408889634

    # Pre-load softmax_lse for all rows in this tile [BLOCK_M]
    safe_rows = tl.minimum(rows, seq_q - 1)
    slse_log2 = tl.load(
        softmax_lse_ptr + slse_base + safe_rows * stride_slse_row
    ) * LOG2E_C

    # ── Load full [BLOCK_M, N_PAD] tile_lse block ──────────────────────
    n_offs = tl.arange(0, N_PAD)           # [N_PAD]
    valid_n = n_offs < N_ACTUAL

    ptrs = (tile_lse_ptr + lse_base
            + rows[:, None]  * stride_lse_row
            + n_offs[None, :] * stride_lse_n)
    lse = tl.load(ptrs,
                  mask=valid_rows[:, None] & valid_n[None, :],
                  other=float('-inf'))

    # ── Reduce [BLOCK_M, N_PAD] → [N_PAD] tile_mass ────────────────────
    row_mass = tl.exp2(lse - slse_log2[:, None])
    row_mass = tl.where(valid_rows[:, None], row_mass, 0.0)
    tile_mass = tl.sum(row_mass, axis=0)   # [N_PAD]

    # ── Normalize + sort + cumsum + threshold → mask ────────────────────
    total     = tl.sum(tile_mass)
    tile_prob = tile_mass / tl.maximum(total, 1e-20)
    tile_prob = tl.where(valid_n, tile_prob, 0.0)

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
    mask   = tl.where(ignore, 0, 1).to(tl.uint8)
    mask   = tl.where(total > 1e-12, mask, 1).to(tl.uint8)

    # ── Write output ─────────────────────────────────────────────────────
    out_offset = pid_bh * Tm * N_ACTUAL + pid_tile * N_ACTUAL + n_offs
    tl.store(mask_out_ptr + out_offset, mask, mask=valid_n)


# Threshold below which the fused single-kernel path is used.
# For N_PAD ≤ _FUSED_N_PAD_THRESHOLD we skip the scratch buffer entirely.
_FUSED_N_PAD_THRESHOLD = 512


def mask_from_stats_fused(
    tile_lse: torch.Tensor,      # [B, H, seq_q, N] float32 — per-row cumulative LSE (log2 domain)
    softmax_lse: torch.Tensor,   # [B, H, seq_q] float32 — full-row LSE (ln domain)
    num_row_tiles_bwd: int,      # Tm = number of backward m-tiles
    kBlockM_bwd: int,            # rows per backward tile
    negl_prob: float,
) -> torch.Tensor:
    """
    Fused mask generation: per-block LSE → uint8 block mask.

    When N_PAD ≤ _FUSED_N_PAD_THRESHOLD (512), uses a single fused kernel that
    keeps tile_mass in registers and avoids scratch buffer allocation + extra
    kernel launch.  For larger N falls back to the two-kernel pipeline.

    Input tile_lse: [B, H, seq_q, N] float32 (log2 domain, per-block LSE,
        stride_row=1 from forward kernel).
    Input softmax_lse: [B, H, seq_q] float32 (ln domain, from FA3 forward).
    Returns uint8 mask of shape [B, H, Tm, N].
    """
    B, H, seq_q, N = tile_lse.shape
    BH = B * H
    Tm = num_row_tiles_bwd

    assert tile_lse.dtype == torch.float32, f"tile_lse must be float32, got {tile_lse.dtype}"
    assert softmax_lse.dtype == torch.float32, f"softmax_lse must be float32, got {softmax_lse.dtype}"
    assert softmax_lse.shape[:2] == (B, H), \
        f"softmax_lse batch/head {softmax_lse.shape[:2]} doesn't match tile_lse (B={B}, H={H})"

    # N_PAD must be power of 2 for tl.sort
    N_PAD = _next_power_of_2(N)
    assert N_PAD <= 2048, f"N_PAD={N_PAD} too large for Triton sort (max 2048)" # TODO: Test limit, currently max seq len supported is 256K with kBlockN=128

    stride_lse_b, stride_lse_h, stride_lse_row, stride_lse_n = tile_lse.stride()
    stride_slse_b, stride_slse_h, stride_slse_row = softmax_lse.stride()

    mask_out = torch.empty(BH, Tm, N, device=tile_lse.device, dtype=torch.uint8)
    grid = (BH, Tm)

    if N_PAD <= _FUSED_N_PAD_THRESHOLD:
        # ── Fast path: single fused kernel, no scratch buffer ──────────
        # num_warps: scale down for tiny N to avoid idle threads.
        if N_PAD <= 32:
            num_warps = 1
        elif N_PAD <= 128:
            num_warps = 2
        else:
            num_warps = 4

        _fused_mask_kernel[grid](
            tile_lse, softmax_lse, mask_out,
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
            N_PAD=N_PAD,
            BLOCK_M=kBlockM_bwd,
            Tm=Tm,
            num_warps=num_warps,
        )
    else:
        # ── Fallback: two-kernel pipeline for large N ──────────────────
        scratch = torch.empty(BH, Tm, N_PAD, device=tile_lse.device, dtype=torch.float32)

        _reduce_tile_mass_kernel[grid](
            tile_lse, softmax_lse, scratch,
            H=H,
            seq_q=seq_q,
            stride_lse_b=stride_lse_b,
            stride_lse_h=stride_lse_h,
            stride_lse_row=stride_lse_row,
            stride_lse_n=stride_lse_n,
            stride_slse_b=stride_slse_b,
            stride_slse_h=stride_slse_h,
            stride_slse_row=stride_slse_row,
            N_ACTUAL=N,
            BLOCK_M=kBlockM_bwd,
            BLOCK_N=min(64, N_PAD),
            N_PAD=N_PAD,
            Tm=Tm,
            num_warps=4,
        )

        _threshold_mask_kernel[grid](
            scratch, mask_out,
            negl_prob,
            N_ACTUAL=N,
            N_PAD=N_PAD,
            Tm=Tm,
            num_warps=4,
        )

    return mask_out.view(B, H, Tm, N)