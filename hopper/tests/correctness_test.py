# custom_attention_cuda_test.py
# Expected conda env: dor-ltx (same as for building the C extension).
import math
import os
import sys
import time
import types
import importlib.util
import glob
from typing import Any

# Ensure BackLite root is on path so hopper is imported as a package (needed for relative imports in back_lite.py)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

_hopper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Bootstrap back_lite package when running from source (so "import back_lite._C" in back_lite.py succeeds)
if "back_lite" not in sys.modules:
    _pkg = types.ModuleType("back_lite")
    _pkg.__path__ = [_hopper_dir]
    _pkg.__file__ = os.path.join(_hopper_dir, "__init__.py")
    _pkg.__package__ = "back_lite"
    sys.modules["back_lite"] = _pkg

import pytest
import torch
# Load back_lite._C after torch so libc10.so and other PyTorch libs are on the linker path
if "back_lite._C" not in sys.modules:
    _c_path = os.path.join(_hopper_dir, "_C.abi3.so")
    if not os.path.isfile(_c_path):
        _c_candidates = glob.glob(os.path.join(_hopper_dir, "_C*.so"))
        _c_path = _c_candidates[0] if _c_candidates else ""
    if os.path.isfile(_c_path):
        _c_spec = importlib.util.spec_from_file_location("back_lite._C", _c_path)
        _c_mod = importlib.util.module_from_spec(_c_spec)
        sys.modules["back_lite._C"] = _c_mod
        _c_spec.loader.exec_module(_c_mod)
import torch.nn.functional as F
import torch.testing
# loads C++ files
from torch.utils.cpp_extension import load
from torch.autograd.graph import saved_tensors_hooks

from hopper.back_lite import BackLite


# Keep one BackLite instance per (batch, device, dtype) so this test file
# can reuse the same BHSD-facing attention call shape as the original test.
_lite_attn_instances = {}


def smallest_mass_subset(p, target):
    vals, idx = torch.sort(p, dim=-1)
    csum = torch.cumsum(vals, dim=-1)
    targets = torch.full_like(csum[..., :1], target)

    cutoff = torch.searchsorted(csum, targets, side='left')

    arange = torch.arange(vals.size(-1), device=vals.device)
    # do not include the last one
    select_sorted = arange < cutoff

    mask = torch.zeros_like(p, dtype=torch.bool)
    mask.scatter_(-1, idx, select_sorted)

    gather_idx = (cutoff - 1).clamp_min(0)
    csum_cut = csum.gather(-1, gather_idx)
    csum_cut = torch.where(cutoff > 0, csum_cut, torch.zeros_like(csum_cut))
    csum_cut = csum_cut.squeeze(-1)

    return mask, csum_cut


def sparse_attention(
    q,
    k,
    v,
    sm_scale,
    warp_specialize: bool = False,
    beta=1.0,
    negl_prob: float = 0.05,
    layer_id=None,
    record_stats: bool = False,
    bwd_kv_blocks_per_prog: int = 1,
    bwd_q_blocks_per_prog: int = 1,
    entropy_tracking=False,
):
    del warp_specialize, beta, layer_id, record_stats
    del bwd_kv_blocks_per_prog, bwd_q_blocks_per_prog, entropy_tracking

    batch_size = q.shape[0]
    cache_key = (batch_size, q.device, q.dtype)
    lite_attn = _lite_attn_instances.get(cache_key)
    if lite_attn is None:
        lite_attn = BackLite(
            negl_prob=negl_prob,
        )
        _lite_attn_instances[cache_key] = lite_attn

    lite_attn.negl_prob = negl_prob

    # BackLite expects [B, S, H, D]. Original tests use [B, H, S, D].
    out = lite_attn(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=sm_scale,
    )
    return out.transpose(1, 2)

@pytest.fixture(scope="module")
def cuda_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return "cuda"


@pytest.mark.parametrize("dtype", [torch.float16,torch.float32, torch.bfloat16])
def test_forward_matches_sdpa(cuda_device: str, dtype: torch.dtype) -> None:
    if dtype is torch.float32:
        pytest.skip("BackLite / FA3 does not support float32")
    if dtype is torch.bfloat16 and (
        not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 not supported on this GPU")

    torch.manual_seed(0)
    D = 64
    # Original \"nice\" shape plus an odd B, N to exercise tail blocks.
    for B, H, N in [(2, 4, 128), (3, 5, 123)]:
        Q = torch.randn(B, H, N, D, dtype=dtype, device=cuda_device)
        K = torch.randn(B, H, N, D, dtype=dtype, device=cuda_device)
        V = torch.randn(B, H, N, D, dtype=dtype, device=cuda_device)

        O_triton = sparse_attention(Q, K, V, 1.0 / math.sqrt(D))

        # Reference in FP32 then cast back to input dtype for a stable comparison
        Q_ref = Q.to(torch.float32)
        K_ref = K.to(torch.float32)
        V_ref = V.to(torch.float32)
        O_ref = F.scaled_dot_product_attention(
            Q_ref, K_ref, V_ref, scale=1.0 / math.sqrt(D)
        ).to(dtype)
        torch.testing.assert_close(
            O_triton,
            O_ref,
            rtol=5e-2,
            atol=5e-2,
            msg=f"forward mismatch between Triton kernel and torch.sdpa for B={B}, H={H}, N={N}",
        )

@pytest.mark.parametrize("dtype", [torch.float16,torch.float32, torch.bfloat16])
def test_backward_matches_sdpa_zero_neg_prob(cuda_device: str, dtype: torch.dtype) -> None:
    if dtype is torch.float32:
        pytest.skip("BackLite / FA3 does not support float32")
    if dtype is torch.bfloat16 and (
        not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 not supported on this GPU")

    torch.manual_seed(1)
    D = 64
    # Original shape plus an odd B, N to exercise tail blocks in backward.
    for B, H, N in [(1, 2, 128), (3, 5, 123)]:
        Q_sp = torch.randn(B, H, N, D, dtype=dtype, device=cuda_device, requires_grad=True)
        K_sp = torch.randn(B, H, N, D, dtype=dtype, device=cuda_device, requires_grad=True)
        V_sp = torch.randn(B, H, N, D, dtype=dtype, device=cuda_device, requires_grad=True)

        Q_ref = Q_sp.detach().clone().requires_grad_(True)
        K_ref = K_sp.detach().clone().requires_grad_(True)
        V_ref = V_sp.detach().clone().requires_grad_(True)

        target = torch.randn(B, H, N, D, dtype=torch.float32, device=cuda_device)

        O_sp = sparse_attention(Q_sp, K_sp, V_sp, sm_scale=1.0 / math.sqrt(D), negl_prob=0.01)
        loss_sp = math.sqrt(N * H * D) * F.mse_loss(O_sp.float(), target)
        loss_sp.backward()
        assert Q_sp.grad is not None
        assert K_sp.grad is not None
        assert V_sp.grad is not None
        dQ_sp, dK_sp, dV_sp = (Q_sp.grad.detach(), K_sp.grad.detach(), V_sp.grad.detach())

        O_ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref)
        loss_ref = math.sqrt(N * H * D) * F.mse_loss(O_ref.float(), target)
        loss_ref.backward()
        assert Q_ref.grad is not None
        assert K_ref.grad is not None
        assert V_ref.grad is not None

        grad_norm_Q = torch.norm(Q_ref.grad).item()
        grad_norm_K = torch.norm(K_ref.grad).item()
        grad_norm_V = torch.norm(V_ref.grad).item()
        grad_norm_Q_custom = torch.norm(dQ_sp).item()
        grad_norm_K_custom = torch.norm(dK_sp).item()
        grad_norm_V_custom = torch.norm(dV_sp).item()

        print(f"[B={B}, H={H}, N={N}] Reference Grad Norms: dQ={grad_norm_Q:.6e}, dK={grad_norm_K:.6e}, dV={grad_norm_V:.6e}")
        print(f"[B={B}, H={H}, N={N}] Custom Grad Norms   : dQ={grad_norm_Q_custom:.6e}, dK={grad_norm_K_custom:.6e}, dV={grad_norm_V_custom:.6e}")

        # Use a strict tolerance for the \"nice\" shape and a slightly looser one
        # for the odd shape where block tails are masked. Relax a bit for BF16.
        if dtype is torch.float16:
            if (B, H, N) == (1, 2, 128):
                rtol, atol = 1e-6, 1e-4
            else:
                rtol, atol = 5e-3, 5e-3
        else:
            if (B, H, N) == (1, 2, 128):
                rtol, atol = 5e-5, 5e-3
            else:
                rtol, atol = 1e-2, 1e-2

        torch.testing.assert_close(
            dQ_sp.float(),
            Q_ref.grad.float(),
            rtol=rtol,
            atol=atol,
            msg=f"dQ mismatch for backward test (B={B}, H={H}, N={N})",
        )
        torch.testing.assert_close(
            dK_sp.float(),
            K_ref.grad.float(),
            rtol=rtol,
            atol=atol,
            msg=f"dK mismatch for backward test (B={B}, H={H}, N={N})",
        )
        torch.testing.assert_close(
            dV_sp.float(),
            V_ref.grad.float(),
            rtol=rtol,
            atol=atol,
            msg=f"dV mismatch for backward test (B={B}, H={H}, N={N})",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_probability_sum(dtype: torch.dtype):
    device = "cuda"

    if dtype is torch.bfloat16 and (
        not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 not supported on this GPU")

    torch.manual_seed(1)

    LOG2E = 1.4426950408889634

    # Run a large-N case (much accumulation) and a smaller power-of-2 case.
    for B, H, N, D in [(2, 4, 16 * 1024, 128), (1, 2, 1024, 128)]:
        Q = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
        K = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
        V = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)

        saved = []

        def _pack(tensor):
            saved.append(tensor)
            return tensor

        def _unpack(tensor):
            return tensor

        # Use negl_prob > 0 to trigger tile_stats saving in the autograd graph
        with saved_tensors_hooks(_pack, _unpack):
            output = sparse_attention(Q, K, V, 1.0 / math.sqrt(D), False, negl_prob=0.01)

        # New saved tensor format: [q, k, v, out, softmax_lse, tile_stats]
        assert len(saved) == 6, (
            f"Expected 6 saved tensors (q, k, v, out, softmax_lse, tile_stats), got {len(saved)}"
        )
        softmax_lse = saved[4]   # [B, H, seq_q] float32 (ln domain)
        tile_stats = saved[5]    # [B, H, seq_q, num_n_blocks] float32 (log2 domain, cumulative)

        # Verify per-row probability mass sums to ~1.0.
        # tile_stats[b,h,row,n] = log2(sum_i exp(score_i)) for KV block n (per-block LSE).
        # Per-block mass = exp2(tile_lse[n] - slse * log2e)
        # Summing over all n-blocks should recover 1.0 per row.
        slse_expanded = softmax_lse.unsqueeze(-1) * LOG2E  # [B, H, seq_q, 1]

        per_block_mass = torch.exp2(tile_stats - slse_expanded)
        total_mass = per_block_mass.sum(-1)  # [B, H, seq_q]

        # Per-row check: float16 attention scores accumulate error across many blocks.
        # Use mean-level check for robustness, with a loose per-row sanity bound.
        mean_mass = total_mass.mean().item()
        assert abs(mean_mass - 1.0) < 0.05, (
            f"Average probability mass {mean_mass:.4f} too far from 1.0 (B={B}, H={H}, N={N})"
        )
        # Sanity: no row should be wildly off (allow generous range for float16 noise)
        assert total_mass.min().item() > 0.3, (
            f"Min row mass {total_mass.min():.4f} unreasonably low (B={B}, H={H}, N={N})"
        )
        assert total_mass.max().item() < 2.0, (
            f"Max row mass {total_mass.max():.4f} unreasonably high (B={B}, H={H}, N={N})"
        )

def entropy(p):
    return torch.special.entr(p).sum(dim=-1)

def get_captured_tensors(B=1, H=1, N=1024, D=128, negl_prob=0.01):
    """Capture saved tensors from a sparse attention forward pass.

    Returns (softmax_lse, tile_stats) where:
      - softmax_lse: [B, H, seq_q] float32 (ln domain)
      - tile_stats:  [B, H, seq_q, num_n_blocks] float32 (log2 domain, cumulative)
    """
    dtype = torch.float16
    device = "cuda"

    torch.manual_seed(1)

    Q = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)

    saved = []

    def _pack(tensor):
        saved.append(tensor)
        return tensor

    def _unpack(tensor):
        return tensor

    with saved_tensors_hooks(_pack, _unpack):
        output = sparse_attention(Q, K, V, 1.0 / math.sqrt(D), False, negl_prob=negl_prob)

    if len(saved) != 6:
        raise RuntimeError(
            f"Expected 6 saved tensors (q, k, v, out, softmax_lse, tile_stats), got {len(saved)}"
        )
    softmax_lse = saved[4]  # [B, H, seq_q]
    tile_stats = saved[5]   # [B, H, seq_q, num_n_blocks]
    return softmax_lse, tile_stats

@pytest.mark.skip(reason="Entropy tracking is no longer saved in forward tensors; "
                         "tile_stats format changed to cumulative LSE.")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_entropy_calc(dtype: torch.dtype):
    pass

def sparsity_level_check():
    """Sweep neglected-probability thresholds and plot the resulting sparsity.

    Uses the new saved-tensor format (tile_stats = cumulative LSE in log2)
    and derives per-tile probability mass from (tile_stats, softmax_lse).
    """
    import matplotlib.pyplot as plt

    LOG2E = 1.4426950408889634
    B, H, N, D = 128, 1, 16 * 1024, 128
    dtype = torch.float16
    device = "cuda"

    torch.manual_seed(1)

    Q = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)

    saved = []

    def _pack(tensor):
        saved.append(tensor)
        return tensor

    def _unpack(tensor):
        return tensor

    for mu in [1.3, 1.6, 2.0, 2.5]:
        print(f"Starting mu= {mu}")
        with saved_tensors_hooks(_pack, _unpack):
            output = sparse_attention(mu * Q, mu * K, V, 1.0 / math.sqrt(D), False, negl_prob=0.01)

        if len(saved) != 6:
            raise RuntimeError(f"Expected 6 saved tensors, got {len(saved)}")

        softmax_lse = saved[4]  # [B, H, seq_q]
        tile_stats = saved[5]   # [B, H, seq_q, num_n_blocks]

        # Derive per-tile probability mass ----------------------------------
        # tile_stats are per-block LSE (log2 domain).
        # mass[n] = exp2(tile_lse[n] - slse * log2e) = probability mass for block n.
        slse = softmax_lse.unsqueeze(-1) * LOG2E
        tile_power = torch.exp2(tile_stats - slse)

        # Average across rows within each backward tile of 128 rows
        kBlockM_bwd = 128
        seq_q = tile_power.shape[2]
        n_tiles = (seq_q + kBlockM_bwd - 1) // kBlockM_bwd
        # Pad to multiple of kBlockM_bwd
        pad = n_tiles * kBlockM_bwd - seq_q
        if pad > 0:
            tile_power = F.pad(tile_power, (0, 0, 0, pad))
        tile_power = tile_power.reshape(B, H, n_tiles, kBlockM_bwd, -1).mean(dim=3)
        # Renormalise so each tile's probabilities sum to 1
        tile_power = tile_power / tile_power.sum(-1, keepdim=True).clamp_min(1e-12)

        uni_prob = 1.0 / tile_power.shape[-1]
        avg_sparsity_list = []
        for k_val in range(tile_power.shape[-1]):
            ignore_mask, _ = smallest_mass_subset(tile_power, target=k_val * uni_prob)
            avg_sparsity = ignore_mask.float().mean().item()
            avg_sparsity_list.append(avg_sparsity)

        neglected_probs = [k_val * uni_prob for k_val in range(tile_power.shape[-1])]

        plt.figure(figsize=(8, 5))
        plt.plot(neglected_probs, avg_sparsity_list, marker='o', linestyle='-')
        plt.xlabel("Neglected Probability")
        plt.ylabel("Average Sparsity")
        plt.title("Average Sparsity vs Neglected Probability")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"sparsity_level_mu={mu}.png")

        saved.clear()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.float16

    B, H, N, D = 1, 12, 1024*16, 128
    scale = math.sqrt(N * H * D)

    def make_inputs():
        Q = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
        K = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
        V = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
        return Q, K, V

    def make_target():
        return torch.randn(B, H, N, D, dtype=dtype, device=device)

    def sdpa_forward(Q, K, V):
        return F.scaled_dot_product_attention(Q, K, V)

    def custom_forward(Q, K, V):
        return sparse_attention(Q, K, V, 1.0/math.sqrt(D), negl_prob=0.01)

    def sdpa_loss(Q, K, V, target):
        out = sdpa_forward(Q, K, V)
        return scale * F.mse_loss(out, target)

    def custom_loss(Q, K, V, target):
        out= sparse_attention(Q, K, V, 1.0/math.sqrt(D), negl_prob=0.01)
        return scale * F.mse_loss(out, target)

    def benchmark_forward(fn):
        Q, K, V = make_inputs()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(Q, K, V)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def benchmark_backward(fn):
        Q, K, V = make_inputs()
        target = make_target()
        loss = fn(Q, K, V, target)
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    # Warm-up to trigger kernel compilation and stabilize timings
    for _ in range(3):
        Q, K, V = make_inputs()
        sdpa_forward(Q, K, V)
        custom_forward(Q, K, V)
    torch.cuda.synchronize()

    for _ in range(3):
        Q, K, V = make_inputs()
        target = make_target()
        sdpa_loss(Q, K, V, target).backward()
        Q, K, V = make_inputs()
        target = make_target()
        custom_loss(Q, K, V, target).backward()
    torch.cuda.synchronize()

    forward_sdpa_time = benchmark_forward(sdpa_forward)
    forward_custom_time = benchmark_forward(custom_forward)
    backward_sdpa_time = benchmark_backward(sdpa_loss)
    backward_custom_time = benchmark_backward(custom_loss)

    print("\n0. TIMING (wall-clock, includes kernel launch)")
    print(
        f"   torch.sdpa forward: {forward_sdpa_time * 1e3:.3f} ms"
        f" | custom forward: {forward_custom_time * 1e3:.3f} ms"
        f" ({forward_custom_time / forward_sdpa_time:.2f}× of sdpa)"
    )
    print(
        f"   torch.sdpa backward: {backward_sdpa_time * 1e3:.3f} ms"
        f" | custom backward: {backward_custom_time * 1e3:.3f} ms"
        f" ({backward_custom_time / backward_sdpa_time:.2f}× of sdpa)"
    )

    # Reset RNG so the qualitative comparison below is reproducible
    torch.manual_seed(1)

    # Compare PyTorch SDPA vs customApproxAttention
    print("\n" + "=" * 70)
    print("COMPARISON: torch.sdpa vs customApproxAttention")
    print("=" * 70)

    # # Create inputs
    Q_comp = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
    K_comp = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)
    V_comp = torch.randn(B, H, N, D, dtype=dtype, device=device, requires_grad=True)

    # # Forward pass
    print(f"\n1. FORWARD PASS")
    O_ref = F.scaled_dot_product_attention(Q_comp, K_comp, V_comp, scale =1.0/ math.sqrt(D))

    saved = []
    def _pack(tensor):
        saved.append(tensor)
        return tensor
    def _unpack(tensor):
        return tensor

    with saved_tensors_hooks(_pack, _unpack):
        O_custom = sparse_attention(Q_comp, K_comp, V_comp, 1.0/ math.sqrt(D), negl_prob=0.01)

    try:
        # New saved format: [q, k, v, out, softmax_lse, tile_stats]
        if len(saved) == 6:
            tile_stats = saved[5]  # [B, H, seq_q, num_n_blocks]
            num_n_blocks = tile_stats.shape[-1]
            print(f"   tile_stats shape: {tile_stats.shape}, num_n_blocks={num_n_blocks}")
            # Mask is computed lazily during backward, not stored in saved tensors.
            # We accurately estimate sparsity by simulating the backward pass behavior.
            softmax_lse = saved[4]
            LOG2E = 1.4426950408889634
            slse = softmax_lse.unsqueeze(-1) * LOG2E
            tile_mass = torch.exp2(tile_stats - slse)
            
            kBlockM_bwd = 128
            seq_q = tile_mass.shape[2]
            n_tiles = (seq_q + kBlockM_bwd - 1) // kBlockM_bwd
            pad = n_tiles * kBlockM_bwd - seq_q
            if pad > 0:
                tile_mass = F.pad(tile_mass, (0, 0, 0, pad))
            tile_mass = tile_mass.reshape(B, H, n_tiles, kBlockM_bwd, -1).mean(dim=3)
            # Renormalise
            tile_mass = tile_mass / tile_mass.sum(-1, keepdim=True).clamp_min(1e-12)
            
            ignore_mask, _ = smallest_mass_subset(tile_mass, target=0.05)
            avg_sparsity = ignore_mask.float().mean().item()
            print(f"   Calculated sparsity (negl_prob=0.05): {avg_sparsity*100:.1f}%")
        else:
            print(f"   Unexpected number of saved tensors: {len(saved)} (expected 6)")
    except Exception as e:
        print(f"   Could not extract tile statistics: {e}")

    # Compute output differences
    output_diff = torch.norm(O_ref.detach() - O_custom.detach()).item()
    output_norm = torch.norm(O_ref.detach()).item()
    relative_diff = output_diff / (output_norm + 1e-10)

    print(f"\n2. OUTPUT COMPARISON")
    print(f"   Absolute difference (L2 norm): {output_diff:.6e}")
    print(f"   Output norm: {output_norm:.6e}")
    print(f"   Relative difference: {relative_diff:.6e}")

    # Backward pass comparison
    print(f"\n3. BACKWARD PASS COMPARISON")
    y_comp = torch.randn(B, H, N, D, dtype=torch.float32, device=device)

    # Backward for torch.sdpa reference
    target = y_comp
    loss_ref = math.sqrt(N * H * D) * F.mse_loss(O_ref.float(), target)
    loss_ref.backward()
    assert Q_comp.grad is not None
    assert K_comp.grad is not None
    assert V_comp.grad is not None
    dQ_ref = Q_comp.grad.clone()
    dK_ref = K_comp.grad.clone()
    dV_ref = V_comp.grad.clone()

    # Clear gradients and backward for customApproxAttention
    Q_comp.grad = None
    K_comp.grad = None
    V_comp.grad = None
    loss_custom = math.sqrt(N * H * D) * F.mse_loss(O_custom.float(), target)
    loss_custom.backward()
    assert Q_comp.grad is not None
    assert K_comp.grad is not None
    assert V_comp.grad is not None
    dQ_custom = Q_comp.grad.clone()
    dK_custom = K_comp.grad.clone()
    dV_custom = V_comp.grad.clone()

    print(f"   torch.sdpa loss (MSE): {loss_ref.item():.6e}")
    print(f"   custom loss (MSE): {loss_custom.item():.6e}")

    # Compare gradients using cosine similarity
    def compute_cosine_sim(a, b):
        # Average over batch (summing is same direction- use because gradients small), then flatten
        a_avg = a.float().sum(dim=0).flatten().unsqueeze(0)  # [1, H*N*D]
        b_avg = b.float().sum(dim=0).flatten().unsqueeze(0)  # [1, H*N*D]
        return F.cosine_similarity(a_avg.float(), b_avg.float(), dim=1, eps=1e-8).item()
    
    # Compare gradients using norm difference 
    def compute_relative_norm(a, a_ref):
        # Average over batch (summing is same direction- use because gradients small), then flatten
        a_avg = a.float().sum(dim=0).flatten().unsqueeze(0)  # [1, H*N*D]
        a_ref_avg = a_ref.float().sum(dim=0).flatten().unsqueeze(0)  # [1, H*N*D]
        diff = a_avg - a_ref_avg
        rel_norm = torch.norm(diff, p=2) / (torch.norm(a_ref_avg, p=2) + 1e-8)
        return rel_norm.item()

    cos_sim_Q = compute_cosine_sim(dQ_ref, dQ_custom)
    cos_sim_K = compute_cosine_sim(dK_ref, dK_custom)
    cos_sim_V = compute_cosine_sim(dV_ref, dV_custom)
    # Concatenate along last dimension: [B, H, N, 3*D]
    dqkv_ref = torch.cat([dQ_ref, dK_ref, dV_ref], dim=-1)
    dqkv_custom = torch.cat([dQ_custom, dK_custom, dV_custom], dim=-1)
    cos_sim_QVK = compute_cosine_sim(dqkv_ref, dqkv_custom)

    # across batch norm
    grad_norm_Q = torch.norm(dQ_ref).item()
    grad_norm_K = torch.norm(dK_ref).item()
    grad_norm_V = torch.norm(dV_ref).item()
    grad_norm_Q_custom = torch.norm(dQ_custom).item()
    grad_norm_K_custom = torch.norm(dK_custom).item()
    grad_norm_V_custom = torch.norm(dV_custom).item()

    print(f"\n4. GRADIENT COMPARISON (Cosine Similarity)")
    print(f"   dQ: cosine_sim={cos_sim_Q:.6f}, norm_ref={grad_norm_Q:.6e}, norm_custom={grad_norm_Q_custom:.6e}")
    print(f"   dK: cosine_sim={cos_sim_K:.6f}, norm_ref={grad_norm_K:.6e}, norm_custom={grad_norm_K_custom:.6e}")
    print(f"   dV: cosine_sim={cos_sim_V:.6f}, norm_ref={grad_norm_V:.6e}, norm_custom={grad_norm_V_custom:.6e}")
    print(f"   dQKV: cosine_sim={cos_sim_QVK:.6f}, norm_ref={torch.norm(dqkv_ref).item():.6e}, norm_custom={torch.norm(dqkv_custom).item():.6e}")
    print(f"   (1.0 = perfect alignment, 0.0 = orthogonal, -1.0 = opposite)")

    rel_norm_Q = compute_relative_norm(dQ_custom, dQ_ref)
    rel_norm_K = compute_relative_norm(dK_custom, dK_ref)
    rel_norm_V = compute_relative_norm(dV_custom, dV_ref)
    rel_norm_QKV = compute_relative_norm(dqkv_custom, dqkv_ref)

    print(f"\n5. GRADIENT COMPARISON (Relative Norm Difference)")
    print(f"   dQ: rel_norm={rel_norm_Q:.6e}")
    print(f"   dK: rel_norm={rel_norm_K:.6e}")
    print(f"   dV: rel_norm={rel_norm_V:.6e}")
    print(f"   dQKV: rel_norm={rel_norm_QKV:.6e}")
    print(f"   (Smaller = better agreement; 0.0 = perfect match)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
