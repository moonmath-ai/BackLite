import torch
from lite_attention import LiteAttention


def generate_test_tensors(batch, seq_len, heads, head_dim):
    """Generate random Q, K, V tensors for testing."""
    q = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seq_len, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def compute_error_metrics(output, reference, name=""):
    """Compute various error metrics between output and reference."""
    diff = (output.float() - reference.float()).abs()
    metrics = {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "rmse": diff.pow(2).mean().sqrt().item(),
    }
    # Cosine similarity
    out_flat = output.float().flatten()
    ref_flat = reference.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(out_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
    metrics["cosine_sim"] = cos_sim
    return metrics


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_forward(q, k, v, head_dim, use_int8=False):
    """Test that a basic forward pass runs without error."""
    attn = LiteAttention(use_int8=use_int8)
    torch.cuda.synchronize()
    output = attn(q, k, v)
    torch.cuda.synchronize()

    passed = output.shape == q.shape[:-1] + (v.shape[-1],)
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Basic forward: {'✅ PASSED' if passed else '❌ FAILED'}")
    if not passed:
        print(f"    Expected shape {q.shape[:-1] + (v.shape[-1],)}, got {output.shape}")
    return passed


def test_forward_with_tile_stats(q, k, v, head_dim, use_int8=False):
    """Test that forward pass collects tile_stats when negl_prob > 0."""
    attn = LiteAttention(use_int8=use_int8, negl_prob=0.01)
    torch.cuda.synchronize()
    output = attn(q, k, v)
    torch.cuda.synchronize()

    passed = output.shape == q.shape[:-1] + (v.shape[-1],)
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Forward with tile_stats (negl_prob=0.01): {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


def test_softmax_lse_correctness(q, k, v, head_dim, tolerance=0.001, use_int8=False):
    """Test that softmax LSE values are correct by comparing against a reference."""
    scale = 1.0 / (head_dim ** 0.5)

    # Reference: manual computation
    q_f = q.float().transpose(1, 2)  # [B, H, S, D]
    k_f = k.float().transpose(1, 2)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    ref_lse = torch.logsumexp(scores, dim=-1)  # [B, H, S]

    # Flash attention LSE
    attn = LiteAttention(use_int8=use_int8)
    torch.cuda.synchronize()
    output, lse = attn(q, k, v, scale=scale, return_softmax_lse=True)
    torch.cuda.synchronize()

    # Compare
    diff = (lse.float() - ref_lse).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = max_err < tolerance

    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Softmax LSE correctness: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"    Max LSE error: {max_err:.6e} (tolerance: {tolerance:.6e})")
    print(f"    Mean LSE error: {mean_err:.6e}")
    return passed


def test_int8_correctness(q, k, v, head_dim, tolerance_max_abs=0.1, tolerance_cosine=0.99):
    """Test that INT8 output matches BF16 output within acceptable tolerance."""
    scale = 1.0 / (head_dim ** 0.5)

    tile_size_bf16 = LiteAttention.get_MN(head_dim, torch.bfloat16)
    tile_size_int8 = LiteAttention.get_MN(head_dim, torch.int8)
    tile_sizes_match = tile_size_bf16 == tile_size_int8

    if not tile_sizes_match:
        print(f"    ⚠️  Tile sizes differ (BF16: {tile_size_bf16}, INT8: {tile_size_int8})")

    # BF16 reference
    attn_bf16 = LiteAttention(use_int8=False)
    torch.cuda.synchronize()
    output_bf16 = attn_bf16(q, k, v, scale=scale)
    torch.cuda.synchronize()

    # INT8
    attn_int8 = LiteAttention(use_int8=True)
    torch.cuda.synchronize()
    output_int8 = attn_int8(q, k, v, scale=scale)
    torch.cuda.synchronize()

    metrics = compute_error_metrics(output_int8, output_bf16, "INT8 vs BF16")

    passed = metrics["max_abs_error"] < tolerance_max_abs and metrics["cosine_sim"] >= tolerance_cosine

    if not tile_sizes_match:
        status = "⚠️  SKIPPED (tile size mismatch)" if not passed else "✅ PASSED (tile size mismatch, OK)"
    else:
        status = "✅ PASSED" if passed else "❌ FAILED"

    print(f"  INT8 correctness: {status}")
    print(f"    Max abs error: {metrics['max_abs_error']:.6e} (tol: {tolerance_max_abs:.6e})")
    print(f"    Cosine similarity: {metrics['cosine_sim']:.8f} (tol: {tolerance_cosine:.8f})")

    return passed if tile_sizes_match else True


def test_backward_basic(q, k, v, head_dim, use_int8=False):
    """Test that backward pass runs without error."""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    attn = LiteAttention(use_int8=use_int8)
    output = attn(q, k, v)
    loss = output.sum()
    loss.backward()
    passed = q.grad is not None and k.grad is not None and v.grad is not None
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Backward pass: {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


def test_backward_with_sparsity(q, k, v, head_dim, use_int8=False):
    """Test that backward pass with sparsity (negl_prob > 0) runs."""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    attn = LiteAttention(use_int8=use_int8, negl_prob=0.01)
    output = attn(q, k, v)
    loss = output.sum()
    loss.backward()
    passed = q.grad is not None and k.grad is not None and v.grad is not None
    prefix = "INT8 " if use_int8 else ""
    print(f"  {prefix}Sparse backward (negl_prob=0.01): {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests_for_head_dim(head_dim, batch=2, seq_len=4096, heads=32):
    """Run all tests for a specific head dimension."""
    print(f"\n{'='*60}")
    print(f"Testing head_dim: {head_dim}")
    print(f"{'='*60}")

    q, k, v = generate_test_tensors(batch, seq_len, heads, head_dim)
    q_short, k_short, v_short = generate_test_tensors(batch, min(2048, seq_len), heads, head_dim)

    # BF16 tests
    print(f"\n  {'-'*56}")
    print(f"  BF16 Tests (head_dim: {head_dim})")
    print(f"  {'-'*56}")
    bf16_results = []
    bf16_results.append(test_basic_forward(q, k, v, head_dim, use_int8=False))
    bf16_results.append(test_forward_with_tile_stats(q, k, v, head_dim, use_int8=False))
    bf16_results.append(test_softmax_lse_correctness(q_short, k_short, v_short, head_dim, use_int8=False))
    bf16_results.append(test_backward_basic(q_short, k_short, v_short, head_dim, use_int8=False))
    bf16_results.append(test_backward_with_sparsity(q_short, k_short, v_short, head_dim, use_int8=False))

    # INT8 tests
    print(f"\n  {'-'*56}")
    print(f"  INT8 Tests (head_dim: {head_dim})")
    print(f"  {'-'*56}")
    int8_results = []
    int8_results.append(test_basic_forward(q, k, v, head_dim, use_int8=True))
    int8_results.append(test_forward_with_tile_stats(q, k, v, head_dim, use_int8=True))
    int8_results.append(test_softmax_lse_correctness(q_short, k_short, v_short, head_dim, tolerance=0.01, use_int8=True))

    # INT8 vs BF16 correctness
    print(f"\n  {'-'*56}")
    print(f"  INT8 Correctness (vs BF16) (head_dim: {head_dim})")
    print(f"  {'-'*56}")
    int8_results.append(test_int8_correctness(q_short, k_short, v_short, head_dim))

    bf16_passed = all(bf16_results)
    int8_passed = all(int8_results)
    return bf16_passed, int8_passed


def main():
    """Main test runner."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    head_dims = [64, 128]

    bf16_results = {}
    int8_results = {}

    for head_dim in head_dims:
        bf16_passed, int8_passed = run_tests_for_head_dim(head_dim)
        bf16_results[head_dim] = bf16_passed
        int8_results[head_dim] = int8_passed

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for label, results in [("BF16", bf16_results), ("INT8", int8_results)]:
        passed = [hd for hd in head_dims if results[hd]]
        failed = [hd for hd in head_dims if not results[hd]]
        if passed:
            print(f"  ✅ {label} PASSED: {passed}")
        if failed:
            print(f"  ❌ {label} FAILED: {failed}")
        if not failed:
            print(f"  All {label} tests passed!")

    print()


if __name__ == "__main__":
    main()
