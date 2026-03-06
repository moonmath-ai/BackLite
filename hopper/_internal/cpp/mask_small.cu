#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_check.h"

namespace {

constexpr float kLog2E = 1.4426950408889634f;
constexpr int kMaxSmallMaskCols = 32;

__global__ void block_mask_small_kernel(
    const float* __restrict__ tile_stats,
    const float* __restrict__ softmax_lse,
    uint8_t* __restrict__ mask,
    int64_t stride_stats_b,
    int64_t stride_stats_h,
    int64_t stride_stats_t,
    int64_t stride_stats_n,
    int64_t stride_lse_b,
    int64_t stride_lse_h,
    int64_t stride_lse_t,
    int64_t stride_mask_b,
    int64_t stride_mask_h,
    int64_t stride_mask_n,
    int64_t stride_mask_t,
    int seq_q,
    int num_col_tiles,
    int kBlockM,
    float negl_prob
) {
    int n = threadIdx.x;
    int m_tile = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int row_start = m_tile * kBlockM;

    __shared__ float mass[kMaxSmallMaskCols];
    __shared__ float prob[kMaxSmallMaskCols];
    __shared__ float sorted[kMaxSmallMaskCols];
    __shared__ float total_prob;
    __shared__ float threshold;
    __shared__ int ties_to_ignore;

    if (n < num_col_tiles) {
        float mass_n = 0.0f;
        int64_t stats_base = int64_t(b) * stride_stats_b + int64_t(h) * stride_stats_h + int64_t(n) * stride_stats_n;
        int64_t lse_base = int64_t(b) * stride_lse_b + int64_t(h) * stride_lse_h;
        #pragma unroll
        for (int row = 0; row < kBlockM; ++row) {
            int q = row_start + row;
            float tile_lse = tile_stats[stats_base + int64_t(q) * stride_stats_t];
            float row_lse = softmax_lse[lse_base + int64_t(q) * stride_lse_t];
            mass_n += exp2f(tile_lse - row_lse * kLog2E);
        }
        mass[n] = mass_n;
    }
    if (n == 0) {
        total_prob = 0.0f;
        threshold = -INFINITY;
        ties_to_ignore = 0;
        for (int i = 0; i < num_col_tiles; ++i) {
            total_prob += mass[i];
        }
        if (total_prob > 1e-12f) {
            for (int i = 0; i < num_col_tiles; ++i) {
                prob[i] = mass[i] / total_prob;
                sorted[i] = prob[i];
            }
            for (int i = 1; i < num_col_tiles; ++i) {
                float x = sorted[i];
                int j = i - 1;
                while (j >= 0 && sorted[j] > x) {
                    sorted[j + 1] = sorted[j];
                    --j;
                }
                sorted[j + 1] = x;
            }
            float csum = 0.0f;
            int ignore_count = 0;
            for (int i = 0; i < num_col_tiles; ++i) {
                csum += sorted[i];
                if (csum < negl_prob) {
                    ++ignore_count;
                    threshold = sorted[i];
                } else {
                    break;
                }
            }
            int strict_count = 0;
            for (int i = 0; i < num_col_tiles; ++i) {
                strict_count += prob[i] < threshold;
            }
            ties_to_ignore = ignore_count - strict_count;
        } else {
            for (int i = 0; i < num_col_tiles; ++i) {
                prob[i] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (n < num_col_tiles) {
        uint8_t keep = 1;
        if (total_prob > 1e-12f) {
            if (prob[n] < threshold) {
                keep = 0;
            } else if (prob[n] == threshold) {
                int tie_rank = 0;
                for (int i = 0; i <= n; ++i) {
                    tie_rank += prob[i] == threshold;
                }
                if (tie_rank <= ties_to_ignore) {
                    keep = 0;
                }
            }
        }
        int64_t out_idx = int64_t(b) * stride_mask_b
            + int64_t(h) * stride_mask_h
            + int64_t(n) * stride_mask_n
            + int64_t(m_tile) * stride_mask_t;
        mask[out_idx] = keep;
    }
}

}  // namespace

at::Tensor block_mask_small(const at::Tensor& tile_stats, const at::Tensor& softmax_lse, int64_t kBlockM, double negl_prob) {
    TORCH_CHECK(tile_stats.is_cuda(), "tile_stats must be on CUDA");
    TORCH_CHECK(softmax_lse.is_cuda(), "softmax_lse must be on CUDA");
    TORCH_CHECK(tile_stats.dtype() == at::kFloat, "tile_stats must be float32");
    TORCH_CHECK(softmax_lse.dtype() == at::kFloat, "softmax_lse must be float32");
    TORCH_CHECK(tile_stats.dim() == 4, "tile_stats must have shape [B, H, seq_q, N]");
    TORCH_CHECK(softmax_lse.dim() == 3, "softmax_lse must have shape [B, H, seq_q]");

    auto sizes = tile_stats.sizes();
    int64_t batch = sizes[0];
    int64_t heads = sizes[1];
    int64_t seq_q = sizes[2];
    int64_t num_col_tiles = sizes[3];
    TORCH_CHECK(softmax_lse.size(0) == batch && softmax_lse.size(1) == heads && softmax_lse.size(2) == seq_q,
        "softmax_lse shape must match tile_stats [B, H, seq_q]");
    TORCH_CHECK(kBlockM > 0 && seq_q % kBlockM == 0, "small mask path requires seq_q divisible by kBlockM");
    TORCH_CHECK(num_col_tiles > 0 && num_col_tiles <= kMaxSmallMaskCols, "small mask path supports 1 <= N <= 32");

    c10::cuda::CUDAGuard device_guard(tile_stats.device());
    auto mask = at::empty({batch, heads, num_col_tiles, seq_q / kBlockM}, tile_stats.options().dtype(at::kByte));
    dim3 grid(seq_q / kBlockM, heads, batch);
    auto stream = at::cuda::getCurrentCUDAStream(tile_stats.device().index());
    block_mask_small_kernel<<<grid, kMaxSmallMaskCols, 0, stream>>>(
        tile_stats.data_ptr<float>(),
        softmax_lse.data_ptr<float>(),
        mask.data_ptr<uint8_t>(),
        tile_stats.stride(0),
        tile_stats.stride(1),
        tile_stats.stride(2),
        tile_stats.stride(3),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(2),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        mask.stride(3),
        seq_q,
        num_col_tiles,
        static_cast<int>(kBlockM),
        static_cast<float>(negl_prob)
    );
    CHECK_CUDA_KERNEL_LAUNCH();
    return mask;
}
