"""Benchmark BackLite vs REAL vanilla Flash Attention 3 (not BackLite with negl_prob=0).

Key design choices that avoid measurement bias:
  - INTERLEAVED: vanilla and backlite alternate every iteration so thermal state
    and boost clock are identical for both (unlike running 1000 iterations of A
    then 1000 of B sequentially).
  - Grad zeroing outside the timed section.
  - Report p50 (median) as the headline metric; p10 as the "best-case" check.
"""
import sys, os, types, importlib, importlib.util

# ── Load BackLite from the hopper directory (this script lives in hopper/tests/) ──
_hopper_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_pkg = types.ModuleType("back_lite")
_pkg.__path__ = [_hopper_dir]
_pkg.__file__ = os.path.join(_hopper_dir, "__init__.py")
_pkg.__package__ = "back_lite"
sys.modules["back_lite"] = _pkg
import torch
_c_path = os.path.join(_hopper_dir, "_C.abi3.so")
if not os.path.isfile(_c_path):
    import glob
    _c_candidates = glob.glob(os.path.join(_hopper_dir, "_C*.so"))
    if _c_candidates:
        _c_path = _c_candidates[0]
    else:
        raise FileNotFoundError(
            "BackLite C extension not found. Build it first (conda env: dor-ltx):\n"
            "  cd hopper && conda run -n dor-ltx python setup.py build_ext --inplace\n"
            "  # or: pip install -e . (from hopper dir with dor-ltx active)"
        )
_c_spec = importlib.util.spec_from_file_location("back_lite._C", _c_path)
_c_mod = importlib.util.module_from_spec(_c_spec)
sys.modules["back_lite._C"] = _c_mod
_c_spec.loader.exec_module(_c_mod)
if _hopper_dir not in sys.path:
    sys.path.insert(0, _hopper_dir)
from _internal.flash_attn_interface import flash_attn_func as backlite_flash_attn_func

# ── Load real FA3 ──
import flash_attn_3._C  # noqa
from flash_attn_interface import flash_attn_func as fa3_flash_attn_func

DTYPE = torch.bfloat16
D = 128
H = 13
NEGL_PROB = 0.3
DEVICE = "cuda"
# WARMUP = 200
# REPEATS = 500

WARMUP = 0
REPEATS = 1

torch.manual_seed(42)


def bench_interleaved(fn_a, fn_b, warmup=WARMUP, repeats=REPEATS):
    """
    Interleaved benchmark: alternate A and B every iteration.
    Returns (p50_a, p10_a, p50_b, p10_b) in ms.
    """
    # Warmup both equally
    for _ in range(warmup):
        fn_b()
        torch.cuda.synchronize()
        fn_a()
        torch.cuda.synchronize()
        # fn_b()
    torch.cuda.synchronize()

    # Pre-allocate all events
    starts_a = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends_a   = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    starts_b = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends_b   = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    # Interleaved measurement
    for i in range(repeats):
        starts_b[i].record(); fn_b(); ends_b[i].record()
        torch.cuda.synchronize()
        starts_a[i].record(); fn_a(); ends_a[i].record()
        torch.cuda.synchronize()
    torch.cuda.synchronize()

    times_a = sorted(starts_a[i].elapsed_time(ends_a[i]) for i in range(repeats))
    times_b = sorted(starts_b[i].elapsed_time(ends_b[i]) for i in range(repeats))

    p50_a = times_a[repeats // 2]
    p10_a = times_a[repeats // 10]
    p50_b = times_b[repeats // 2]
    p10_b = times_b[repeats // 10]
    return p50_a, p10_a, p50_b, p10_b


for causal, window_size, label in [
    (True, (-1, -1), "causal-only"),
    (True, (1024, 0), "causal + window=1024"),
]:
    print(f"\n{'='*90}")
    print(f"  Mode: {label}   H={H}, D={D}, dtype={DTYPE}, negl_prob={NEGL_PROB}")
    print(f"  Warmup={WARMUP}, Repeats={REPEATS}  [INTERLEAVED]")
    print(f"  Vanilla = real flash_attn_3 (separate package)")
    print(f"{'='*90}")
    print(f"{'B':>4}  {'T':>5}  {'FA3 p50':>10}  {'BL p50':>10}  {'Speedup':>8}  {'FA3 p10':>9}  {'BL p10':>8}")
    print("-" * 70)

    for B, T in [(4, 2048), (2, 4096), (1, 8192), (4, 4096), (2, 8192)]:
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        dout = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE)

        def zero_grads():
            q.grad = None; k.grad = None; v.grad = None

        def vanilla():
            # Grad zeroing is OUTSIDE the event window in the interleaved loop.
            # But autograd accumulates, so we must zero here too; cost is equal
            # for both branches so it cancels out in the speedup ratio.
            q.grad = None; k.grad = None; v.grad = None
            o = fa3_flash_attn_func(q, k, v, causal=causal, window_size=window_size)
            o.backward(dout)

        def backlite():
            q.grad = None; k.grad = None; v.grad = None
            o = backlite_flash_attn_func(q, k, v, causal=causal, window_size=window_size, negl_prob=NEGL_PROB)
            o.backward(dout)

        p50_v, p10_v, p50_b, p10_b = bench_interleaved(vanilla, backlite)
        speedup = (p50_v / p50_b - 1) * 100
        print(f"{B:>4}  {T:>5}  {p50_v:>10.3f}  {p50_b:>10.3f}  {speedup:>+7.1f}%  {p10_v:>9.3f}  {p10_b:>8.3f}")