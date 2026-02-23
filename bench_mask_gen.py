#!/usr/bin/env python3
"""
Benchmark: Custom attention forward (with _generate_mask_from_stats_mass) vs Flash Attention.

Measures and compares:
  1. Flash Attention 2 (from flash_attn package) - baseline
  2. Custom FA3 (negl_prob=0) - no mask generation overhead
  3. Custom FA3 (negl_prob>0) - includes mask generation, with timing breakdown

Usage:
    conda activate train_env
    python bench_mask_gen.py
"""

import torch
import time
import statistics

from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_fa2
from lite_attention._internal.flash_attn_interface import (
    flash_attn_func as flash_attn_fa3,
    get_last_mask_gen_time_ms,
)


def bench(fn, *args, warmup=10, repeats=30, **kwargs):
    """Benchmark using CUDA events. Returns list of times in ms."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def bench_with_mask_timing(fn, *args, warmup=10, repeats=30, **kwargs):
    """Benchmark and also capture internal mask-generation time per iteration."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    total_times = []
    mask_gen_times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        total_times.append(start.elapsed_time(end))
        mg = get_last_mask_gen_time_ms()
        if mg is not None:
            mask_gen_times.append(mg)
    return total_times, mask_gen_times


def fmt(times):
    m = statistics.mean(times)
    s = statistics.stdev(times) if len(times) > 1 else 0.0
    return m, s


def main():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    warmup = 10
    repeats = 30

    configs = [
        # (batch, seqlen, nheads, headdim)
        (2, 4096, 32, 128),
        (2, 8192, 32, 128),
        (2, 16384, 32, 128),
        (1, 32768, 32, 128),
        (1, 64*1024, 32, 128),
        (1, 128*1024, 32, 128),
    ]
    negl_probs = [0.01, 0.05, 0.1]

    print("=" * 100)
    print("  Benchmark: Custom Attention Forward (mask gen) vs Flash Attention")
    print("=" * 100)
    print(f"  dtype={dtype}, warmup={warmup} iters, repeats={repeats} iters")
    print()
    print("  'mask_gen' = _generate_mask_from_stats_mass (torch.compiled mask builder)")
    print("  'rest'     = tile_stats alloc + FA3 kernel (kernel writes tile_stats when negl_prob>0)")

    for batch, seqlen, nheads, headdim in configs:
        print(f"\n{'─' * 100}")
        print(f"  batch={batch}  seqlen={seqlen}  nheads={nheads}  headdim={headdim}")
        print(f"{'─' * 100}")

        q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
        v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)

        # --- FA2 baseline ---
        fa2_times = bench(
            flash_attn_fa2, q, k, v, 0.0,
            causal=False, warmup=warmup, repeats=repeats,
        )
        fa2_m, fa2_s = fmt(fa2_times)
        print(f"  FA2 (flash_attn pkg):                 {fa2_m:8.3f} +/- {fa2_s:.3f} ms")
        time.sleep(0.3)

        # --- Custom FA3 without mask generation ---
        fa3_times = bench(
            flash_attn_fa3, q, k, v,
            causal=False, negl_prob=0.0, warmup=warmup, repeats=repeats,
        )
        fa3_m, fa3_s = fmt(fa3_times)
        print(f"  Custom FA3 (negl_prob=0.00):           {fa3_m:8.3f} +/- {fa3_s:.3f} ms")
        time.sleep(0.3)

        # --- Custom FA3 with mask generation ---
        for negl_prob in negl_probs:
            total_times, mask_times = bench_with_mask_timing(
                flash_attn_fa3, q, k, v,
                causal=False, negl_prob=negl_prob,
                warmup=warmup, repeats=repeats,
            )
            total_m, total_s = fmt(total_times)
            if mask_times:
                mask_m, mask_s = fmt(mask_times)
            else:
                mask_m, mask_s = 0.0, 0.0
            rest_m = total_m - mask_m
            pct = (mask_m / total_m * 100) if total_m > 0 else 0
            overhead = total_m - fa3_m

            print(f"  Custom FA3 (negl_prob={negl_prob:.2f}):")
            print(f"      total:                            {total_m:8.3f} +/- {total_s:.3f} ms")
            print(f"      mask_gen:                         {mask_m:8.3f} +/- {mask_s:.3f} ms  ({pct:.1f}% of total)")
            print(f"      rest (kernel+alloc):              {rest_m:8.3f} ms")
            print(f"      overhead vs negl_prob=0:          {overhead:+8.3f} ms")
            time.sleep(0.3)

    print(f"\n{'=' * 100}")
    print("  Benchmark complete.")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
