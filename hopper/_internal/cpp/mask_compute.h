#pragma once

#include <torch/types.h>
#include <cuda_runtime.h>

// Compute block sparsity mask from tile_stats (sortless per-tile threshold).
// Replaces the Python/Triton mask generation kernel with a compiled CUDA kernel
// called directly from the C++ backward path.
//
// tile_lse:      [B, H, T, N] float32   — per-row tile LSE (log2 domain)
// softmax_lse:   [B, H, T]   float32   — full-row LSE (ln domain)
// kBlockM_bwd:   rows per backward m-tile
// negl_prob:     threshold for dropping tiles
// stream:        CUDA stream to launch on
//
// Returns: int32 bitmask [B, H, N, num_words]
at::Tensor compute_block_mask_from_tile_stats(
    const at::Tensor& tile_lse,
    const at::Tensor& softmax_lse,
    int kBlockM_bwd,
    float negl_prob,
    cudaStream_t stream);
