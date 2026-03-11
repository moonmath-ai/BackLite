#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace flash {

// Grid: (B*H, N), 32 threads per block (one warp).
// Computes bitmask for the sortless per-tile threshold: drop tile m if
// tile_prob[m] < negl_prob.  Matches the Triton kernel semantics exactly.
__global__ void compute_block_mask_kernel(
    const float* __restrict__ tile_lse,
    const float* __restrict__ softmax_lse,
    int32_t* __restrict__ bitmask_out,
    int H, int seq_q,
    int64_t stride_lse_b, int64_t stride_lse_h,
    int64_t stride_lse_row, int64_t stride_lse_n,
    int64_t stride_slse_b, int64_t stride_slse_h,
    int64_t stride_slse_row,
    float negl_prob,
    int N, int kBlockM, int Tm, int num_words)
{
    int const pid_bh = blockIdx.x;
    int const pid_n  = blockIdx.y;
    int const lane   = threadIdx.x;  // 0..31

    int const b = pid_bh / H;
    int const h = pid_bh % H;
    int64_t const lse_base  = (int64_t)b * stride_lse_b  + (int64_t)h * stride_lse_h;
    int64_t const slse_base = (int64_t)b * stride_slse_b + (int64_t)h * stride_slse_h;

    constexpr float LOG2E = 1.4426950408889634f;

    // Each tile m: sum exp2(tile_lse[row,n] - softmax_lse[row] * LOG2E) over
    // BLOCK_M rows.  Warp lanes split the rows, then warp-reduce.
    float total_mass = 0.0f;

    // Tile masses stored in per-lane scratch; lane 0 accumulates final words.
    // Max Tm we handle in registers: 256 (covers seq_len up to 32K at kBlockM=128).
    constexpr int kMaxTm = 256;
    float my_tile_mass[kMaxTm];

    for (int m = 0; m < Tm && m < kMaxTm; ++m) {
        float lane_sum = 0.0f;
        int row_start = m * kBlockM;
        // Each of the 32 lanes handles every 32nd row
        for (int r = lane; r < kBlockM && (row_start + r) < seq_q; r += 32) {
            int row = row_start + r;
            float lse  = tile_lse[lse_base + (int64_t)row * stride_lse_row
                                  + (int64_t)pid_n * stride_lse_n];
            float slse = softmax_lse[slse_base + (int64_t)row * stride_slse_row];
            float row_mass = exp2f(lse - slse * LOG2E);
            if (row_mass == row_mass)  // filter NaN from -inf - (-inf)
                lane_sum += row_mass;
        }
        // Warp reduce
        for (int offset = 16; offset > 0; offset >>= 1)
            lane_sum += __shfl_xor_sync(0xFFFFFFFF, lane_sum, offset);
        my_tile_mass[m] = lane_sum;
        total_mass += lane_sum;
    }

    // Threshold + bitpack (lane 0 writes)
    if (lane == 0) {
        bool no_attention = (total_mass <= 1e-12f);
        float inv_total = no_attention ? 0.0f : (1.0f / total_mass);
        int64_t out_base = ((int64_t)pid_bh * N + pid_n) * num_words;
        for (int w = 0; w < num_words; ++w) {
            uint32_t word = 0;
            for (int bit = 0; bit < 32; ++bit) {
                int m = w * 32 + bit;
                if (m >= Tm) break;
                float prob = my_tile_mass[m] * inv_total;
                if (no_attention || prob >= negl_prob)
                    word |= (1u << bit);
            }
            bitmask_out[out_base + w] = static_cast<int32_t>(word);
        }
    }
}

inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

} // namespace flash
