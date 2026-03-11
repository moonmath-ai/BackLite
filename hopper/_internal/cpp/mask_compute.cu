#include "mask_compute.h"
#include <cstdint>
#include <algorithm>

namespace {

// Grid: (B*H, N), 32 threads per block (one warp).
// Computes bitmask for the sortless per-tile threshold: drop tile m if
// tile_prob[m] < negl_prob.
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
    int const lane   = threadIdx.x;

    int const b = pid_bh / H;
    int const h = pid_bh % H;
    int64_t const lse_base  = (int64_t)b * stride_lse_b  + (int64_t)h * stride_lse_h;
    int64_t const slse_base = (int64_t)b * stride_slse_b + (int64_t)h * stride_slse_h;

    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int kMaxTm = 256;

    float tile_mass_arr[kMaxTm];
    float total_mass = 0.0f;

    for (int m = 0; m < Tm && m < kMaxTm; ++m) {
        float lane_sum = 0.0f;
        int row_start = m * kBlockM;
        for (int r = lane; r < kBlockM && (row_start + r) < seq_q; r += 32) {
            int row = row_start + r;
            float lse  = tile_lse[lse_base + (int64_t)row * stride_lse_row
                                  + (int64_t)pid_n * stride_lse_n];
            float slse = softmax_lse[slse_base + (int64_t)row * stride_slse_row];
            float rm = exp2f(lse - slse * LOG2E);
            if (rm == rm) lane_sum += rm;  // NaN guard (-inf - -inf)
        }
        // Warp reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            lane_sum += __shfl_xor_sync(0xFFFFFFFF, lane_sum, offset);
        tile_mass_arr[m] = lane_sum;
        total_mass += lane_sum;
    }

    if (lane == 0) {
        bool no_attention = (total_mass <= 1e-12f);
        float inv_total = no_attention ? 0.0f : (1.0f / total_mass);
        int64_t out_base = ((int64_t)pid_bh * N + pid_n) * num_words;
        for (int w = 0; w < num_words; ++w) {
            uint32_t word = 0;
            for (int bit = 0; bit < 32; ++bit) {
                int m = w * 32 + bit;
                if (m >= Tm) break;
                float prob = tile_mass_arr[m] * inv_total;
                if (no_attention || prob >= negl_prob)
                    word |= (1u << bit);
            }
            bitmask_out[out_base + w] = static_cast<int32_t>(word);
        }
    }
}

int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

}  // anonymous namespace


at::Tensor compute_block_mask_from_tile_stats(
    const at::Tensor& tile_lse,
    const at::Tensor& softmax_lse,
    int kBlockM_bwd,
    float negl_prob,
    cudaStream_t stream)
{
    int B = tile_lse.size(0);
    int H = tile_lse.size(1);
    int seq_q = tile_lse.size(2);
    int N = tile_lse.size(3);
    int BH = B * H;
    int Tm = (seq_q + kBlockM_bwd - 1) / kBlockM_bwd;
    int TM_PAD = std::max(32, next_pow2(Tm));
    int num_words = TM_PAD / 32;

    auto bitmask = at::empty({BH, N, num_words},
                             tile_lse.options().dtype(at::kInt));

    dim3 grid(BH, N);
    compute_block_mask_kernel<<<grid, 32, 0, stream>>>(
        tile_lse.data_ptr<float>(),
        softmax_lse.data_ptr<float>(),
        bitmask.data_ptr<int32_t>(),
        H, seq_q,
        tile_lse.stride(0), tile_lse.stride(1),
        tile_lse.stride(2), tile_lse.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1),
        softmax_lse.stride(2),
        negl_prob, N, kBlockM_bwd, Tm, num_words);

    return bitmask.view({B, H, N, num_words});
}
