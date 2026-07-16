#!/usr/bin/env python3
"""
Correctness + benchmark harness for the softmax CUDA extension.

Compares hand-written kernels against torch.softmax (ground truth, computed in
double precision) and, if available, a Triton fused softmax.

Run on a Colab T4 (Runtime > Change runtime type > T4 GPU):
    !python bench.py
"""

import time
import torch
from torch.utils.cpp_extension import load

assert torch.cuda.is_available(), "No CUDA device. Enable a GPU runtime."
DEV = "cuda"
PEAK_GBPS = 320.0  # T4 marketing peak; ~85-90% achievable in bandwidthTest

print(f"Device: {torch.cuda.get_device_name(0)}")
print("Compiling extension (first run takes ~1-2 min)...")
ext = load(
    name="softmax_ext",
    sources=["softmax_ext.cu"],
    extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
    verbose=False,
)
print("OK\n")

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def gt(x, dim):
    """Ground-truth softmax in double precision, cast back to float for compare."""
    return torch.softmax(x.double(), dim=dim)

def report_err(name, out, ref_double):
    o = out.double()
    abs_err = (o - ref_double).abs().max().item()
    rel_err = ((o - ref_double).abs() / (ref_double.abs() + 1e-30)).max().item()
    print(f"  {name:<26} max_abs={abs_err:.3e}  max_rel={rel_err:.3e}")
    return abs_err

def bench_ms(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms per call

def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9

def line(name, ms, nbytes):
    bw = gbps(nbytes, ms)
    print(f"  {name:<26} {ms*1e3:8.1f} us   {bw:7.1f} GB/s   "
          f"({100*bw/PEAK_GBPS:4.0f}% of {PEAK_GBPS:.0f})")

# --------------------------------------------------------------------------- #
# (A) whole-vector, 10M elements
# --------------------------------------------------------------------------- #

def whole_vector():
    print("=" * 78)
    print("(A) WHOLE-VECTOR SOFTMAX  n = 10,000,000")
    print("=" * 78)
    n = 10_000_000
    torch.manual_seed(0)
    x = torch.randn(n, device=DEV, dtype=torch.float32)
    ref = gt(x, dim=0)
    bytes_io = 2 * n * 4  # ideal fused traffic: read once + write once

    print("Correctness (vs double-precision torch.softmax):")
    outs = {
        "custom v0 naive":  ext.wholevec(x, 0),
        "custom v1 fused":  ext.wholevec(x, 1),
        "custom v2 vec4":   ext.wholevec(x, 2),
        "custom v3 online": ext.wholevec(x, 3),
        "torch.softmax":    torch.softmax(x, dim=0),
    }
    for name, o in outs.items():
        report_err(name, o, ref)
    # invariant: sums to 1
    s = outs["custom v3 online"].double().sum().item()
    print(f"  sum(custom v3)            = {s:.6f}  (should be 1.0)\n")

    print(f"Benchmark  (ideal traffic = {bytes_io/1e6:.0f} MB):")
    line("custom v0 naive",  bench_ms(lambda: ext.wholevec(x, 0)), bytes_io)
    line("custom v1 fused",  bench_ms(lambda: ext.wholevec(x, 1)), bytes_io)
    line("custom v2 vec4",   bench_ms(lambda: ext.wholevec(x, 2)), bytes_io)
    line("custom v3 online", bench_ms(lambda: ext.wholevec(x, 3)), bytes_io)
    line("torch.softmax",    bench_ms(lambda: torch.softmax(x, dim=0)), bytes_io)
    print()

# --------------------------------------------------------------------------- #
# (B) row-wise
# --------------------------------------------------------------------------- #

# optional Triton fused softmax (standard tutorial kernel)
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _tl_softmax(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                    BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(in_ptr + row * in_stride + offs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        tl.store(out_ptr + row * out_stride + offs, num / den, mask=mask)

    def triton_softmax(x):
        N, D = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D)
        nw = 4 if BLOCK <= 1024 else (8 if BLOCK <= 2048 else 16)
        _tl_softmax[(N,)](y, x, x.stride(0), y.stride(0), D, BLOCK=BLOCK, num_warps=nw)
        return y

    HAS_TRITON = True
except Exception as ex:
    HAS_TRITON = False
    print(f"(Triton unavailable: {ex})\n")


def row_wise(N, D):
    print("=" * 78)
    print(f"(B) ROW-WISE SOFTMAX  shape = ({N}, {D})  softmax over dim=1")
    print("=" * 78)
    torch.manual_seed(0)
    x = torch.randn(N, D, device=DEV, dtype=torch.float32)
    ref = gt(x, dim=1)
    bytes_io = 2 * N * D * 4

    print("Correctness (vs double-precision torch.softmax):")
    report_err("custom block", ext.rowwise(x, 0), ref)
    if D <= 1024:
        report_err("custom warp", ext.rowwise(x, 1), ref)
    report_err("torch.softmax", torch.softmax(x, dim=1), ref)
    if HAS_TRITON:
        report_err("triton", triton_softmax(x), ref)
    print()

    print(f"Benchmark  (traffic = {bytes_io/1e6:.0f} MB):")
    line("custom block", bench_ms(lambda: ext.rowwise(x, 0)), bytes_io)
    if D <= 1024:
        line("custom warp", bench_ms(lambda: ext.rowwise(x, 1)), bytes_io)
    line("torch.softmax", bench_ms(lambda: torch.softmax(x, dim=1)), bytes_io)
    if HAS_TRITON:
        line("triton", bench_ms(lambda: triton_softmax(x)), bytes_io)
    print()

# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #

def edge_cases():
    print("=" * 78)
    print("EDGE CASES (custom kernels must match torch)")
    print("=" * 78)
    cases = {
        "all-equal (n=1e6)":     torch.full((1_000_000,), 3.14, device=DEV),
        "large +/-100 (n=1e6)":  (torch.randint(0, 2, (1_000_000,), device=DEV).float()*200 - 100),
        "non-mult-of-4 (n=10000003)": torch.randn(10_000_003, device=DEV),
        "single element":        torch.randn(1, device=DEV),
    }
    for name, x in cases.items():
        x = x.float().contiguous()
        ref = gt(x, dim=0)
        ok = True
        for v in (0, 1, 2, 3):
            o = ext.wholevec(x, v).double()
            bad = (o - ref).abs().max().item()
            ok = ok and (bad < 1e-4)
        print(f"  {name:<30} {'PASS' if ok else 'FAIL'}  (max_abs={bad:.2e})")
    print()

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    whole_vector()
    row_wise(16384, 1024)   # warp-per-row sweet spot (~16.8M elems)
    row_wise(4096, 4096)    # block-per-row regime
    edge_cases()
    print("Done. Next: profile the slow kernel with nsys then ncu (see README).")
