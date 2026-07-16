#!/usr/bin/env python3
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import gc
import time

import numpy as np
import torch
from torch.utils.cpp_extension import load

assert torch.cuda.is_available(), "No CUDA device. Enable a GPU runtime."
GPU_NAME = torch.cuda.get_device_name(0)
PEAK_GBPS = 320.0

N_SWEEP = [100_000, 316_228, 1_000_000, 3_162_278,
           10_000_000, 31_622_777, 100_000_000]
CHECK_N = 1_000_000
SCALE_N = 10_000_000
BLOCKS = [40, 80, 160, 320, 640, 1280, 2560, 5120]
REPS = 5

print(f"Device: {GPU_NAME}")
print("Compiling CUDA extension (cached after first run)...")
ext_cu = load(
    name="softmax_cu_ext",
    sources=["softmax_cu_bind.cu"],
    extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
    extra_include_paths=[os.path.abspath(".")],
    verbose=False,
)
print("OK\n")
def timeit(run, sync, reps=REPS, target_s=0.25):
    run(); sync()
    t0 = time.perf_counter(); run(); sync()
    est = time.perf_counter() - t0
    iters = max(3, min(300, int(target_s / max(est, 1e-5))))
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        sync()
        ts.append((time.perf_counter() - t0) / iters)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]

def make_softmax_cu(x_np):
    xt = torch.from_numpy(x_np).cuda()
    h = {}
    return dict(
        run=lambda: h.__setitem__("y", ext_cu.softmax_cu(xt)),
        sync=torch.cuda.synchronize,
        out=lambda: h["y"].cpu().numpy(),
        clean=lambda: (h.clear(), torch.cuda.empty_cache()),
    )


def make_torch(x_np):
    xt = torch.from_numpy(x_np).cuda()
    h = {}
    return dict(
        run=lambda: h.__setitem__("y", torch.softmax(xt, dim=0)),
        sync=torch.cuda.synchronize,
        out=lambda: h["y"].cpu().numpy(),
        clean=lambda: (h.clear(), torch.cuda.empty_cache()),
    )


def make_triton(x_np):
    import triton
    import triton.language as tl

    BLOCK = 4096

    @triton.jit
    def _partial(x_ptr, pm_ptr, pd_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs, mask=offs < n, other=float("-inf"))
        m = tl.max(x, axis=0)
        d = tl.sum(tl.exp(x - m), axis=0)
        tl.store(pm_ptr + pid, m)
        tl.store(pd_ptr + pid, d)

    @triton.jit
    def _normalize(x_ptr, y_ptr, gm_ptr, gd_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        gm = tl.load(gm_ptr)
        gd = tl.load(gd_ptr)
        tl.store(y_ptr + offs, tl.exp(x - gm) / gd, mask=mask)

    xt = torch.from_numpy(x_np).cuda()
    n = xt.numel()
    nb = triton.cdiv(n, BLOCK)
    y = torch.empty_like(xt)
    pm = torch.empty(nb, device="cuda")
    pd = torch.empty(nb, device="cuda")

    def run():
        _partial[(nb,)](xt, pm, pd, n, BLOCK=BLOCK)
        gm = pm.max().reshape(1)
        gd = (pd * torch.exp(pm - gm)).sum().reshape(1)
        _normalize[(nb,)](xt, y, gm, gd, n, BLOCK=BLOCK)

    return dict(
        run=run,
        sync=torch.cuda.synchronize,
        out=lambda: y.cpu().numpy(),
        clean=torch.cuda.empty_cache,
    )


def make_cupy(x_np):
    import cupy as cp
    try:
        from cupyx.scipy.special import softmax as cp_softmax
    except ImportError:
        def cp_softmax(v):
            e = cp.exp(v - v.max())
            return e / e.sum()
    xc = cp.asarray(x_np)
    h = {}
    return dict(
        run=lambda: h.__setitem__("y", cp_softmax(xc)),
        sync=cp.cuda.runtime.deviceSynchronize,
        out=lambda: cp.asnumpy(h["y"]),
        clean=lambda: (h.clear(), cp.get_default_memory_pool().free_all_blocks()),
    )


def make_jax(x_np):
    import jax
    import jax.numpy as jnp
    f = jax.jit(jax.nn.softmax)
    xj = jnp.asarray(x_np)
    h = {}
    return dict(
        run=lambda: h.__setitem__("y", f(xj)),
        sync=lambda: h["y"].block_until_ready(),
        out=lambda: np.asarray(h["y"]),
        clean=h.clear,
    )


def make_tf(x_np):
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass  # already initialized
    xt = tf.constant(x_np)
    h = {}
    if hasattr(tf.test.experimental, "sync_devices"):
        sync = tf.test.experimental.sync_devices
    else:
        def sync():  # fallback: fetching one element forces execution
            _ = float(h["y"][0])
    return dict(
        run=lambda: h.__setitem__("y", tf.nn.softmax(xt)),
        sync=sync,
        out=lambda: h["y"].numpy(),
        clean=h.clear,
    )


IMPLS = [  # fixed order = fixed color assignment in the plots
    ("softmax.cu (ours)", make_softmax_cu),
    ("Triton",            make_triton),
    ("PyTorch",           make_torch),
    ("CuPy",              make_cupy),
    ("JAX (XLA)",         make_jax),
    ("TensorFlow",        make_tf),
]
def make_input(n):
    rng = np.random.default_rng(42)
    x = rng.uniform(-10, 10, n).astype(np.float32)
    x[n // 3] = 25.0
    return x


def ref_softmax64(x_np):
    x = x_np.astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def run_sweep():
    rows, errs = [], {}
    for n in N_SWEEP:
        x_np = make_input(n)
        ref = ref_softmax64(x_np) if n == CHECK_N else None
        for name, factory in IMPLS:
            try:
                impl = factory(x_np)
                med, lo, hi = timeit(impl["run"], impl["sync"])
                if ref is not None:
                    out = impl["out"]().astype(np.float64)
                    errs[name] = np.max(np.abs(out - ref) / (ref + 1e-30))
                impl["clean"]()
            except Exception as e:
                print(f"  {name:<18} n={n:>11,}  SKIP ({type(e).__name__}: {e})")
                continue
            finally:
                gc.collect()
            gbps = 12 * n / med / 1e9
            rows.append(dict(impl=name, n=n, ms=med * 1e3, ms_min=lo * 1e3,
                             ms_max=hi * 1e3, gbps=gbps))
            print(f"  {name:<18} n={n:>11,}  {med*1e3:9.3f} ms  {gbps:7.1f} GB/s")
        del x_np
        gc.collect()
    return rows, errs


SCALING_IMPLS = [  # name must match IMPLS for consistent colors
    ("softmax.cu (ours)", lambda xt, b: ext_cu.softmax_cu(xt, b)),
]


def run_scaling():
    x_np = make_input(SCALE_N)
    xt = torch.from_numpy(x_np).cuda()
    rows = []
    for name, fn in SCALING_IMPLS:
        print(f"  -- {name}")
        for b in BLOCKS:
            med, lo, hi = timeit(lambda fn=fn, b=b: fn(xt, b),
                                 torch.cuda.synchronize)
            gbps = 12 * SCALE_N / med / 1e9
            rows.append(dict(impl=name, blocks=b, ms=med * 1e3, ms_min=lo * 1e3,
                             ms_max=hi * 1e3, gbps=gbps))
            print(f"  blocks={b:>5}  {med*1e3:8.3f} ms  {gbps:7.1f} GB/s")
    del xt
    torch.cuda.empty_cache()
    return rows
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, BASE = "#e1e0d9", "#fcfcfb", "#c3c2b7"
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "X"]


def style_axes(ax):
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(title, xlabel, ylabel):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURF)
    style_axes(ax)
    ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=SEC, fontsize=10)
    ax.set_ylabel(ylabel, color=SEC, fontsize=10)
    return fig, ax


def legend(ax):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=SEC)
    return leg


def plot_sweep(rows, ykey, ylabel, title, fname, logy, peak=None):
    import matplotlib.pyplot as plt
    fig, ax = new_fig(title, "input elements N", ylabel)
    names = [n for n, _ in IMPLS]
    for i, name in enumerate(names):
        pts = [r for r in rows if r["impl"] == name]
        if not pts:
            continue
        xs = [r["n"] for r in pts]
        ys = [r[ykey] for r in pts]
        ax.plot(xs, ys, color=COLORS[i], marker=MARKERS[i], markersize=6,
                linewidth=2, label=name,
                zorder=10 - i)
        if ykey == "ms":
            ax.fill_between(xs, [r["ms_min"] for r in pts],
                            [r["ms_max"] for r in pts],
                            color=COLORS[i], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if peak:
        ax.axhline(peak, color=BASE, linestyle="--", linewidth=1.2)
        ax.text(ax.get_xlim()[0] * 1.3, peak * 1.03,
                f"T4 DRAM peak {peak:.0f} GB/s", color=MUT, fontsize=8)
    legend(ax)
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"wrote {fname}")


def plot_scaling(rows, fname):
    import matplotlib.pyplot as plt
    fig, ax = new_fig(
        f"custom kernels: strong scaling, N = {SCALE_N:,} (T4, 256 threads/block)",
        "thread blocks", "effective GB/s")
    names = [n for n, _ in IMPLS]
    for name, _ in SCALING_IMPLS:
        pts = [r for r in rows if r.get("impl", names[0]) == name]
        if not pts:
            continue
        i = names.index(name)
        xs = [r["blocks"] for r in pts]
        ys = [r["gbps"] for r in pts]
        ax.plot(xs, ys, color=COLORS[i], marker=MARKERS[i], markersize=6,
                linewidth=2, label=name)
    ax.set_xscale("log", base=2)
    ax.axvline(40, color=BASE, linestyle="--", linewidth=1.2)
    ax.text(42, ax.get_ylim()[0] * 1.05 + 1, "1 block/SM (40 SMs)",
            color=MUT, fontsize=8)
    ax.axhline(PEAK_GBPS, color=BASE, linestyle="--", linewidth=1.2)
    ax.text(BLOCKS[0], PEAK_GBPS * 1.02, f"T4 DRAM peak {PEAK_GBPS:.0f} GB/s",
            color=MUT, fontsize=8)
    legend(ax)
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"wrote {fname}")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")
def main():
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs("results", exist_ok=True)

    print("== size sweep ==")
    rows, errs = run_sweep()
    print("\n== strong scaling (custom kernels) ==")
    srows = run_scaling()

    write_csv("results/sweep.csv", rows)
    write_csv("results/scaling.csv", srows)

    plot_sweep(rows, "ms", "latency (ms, log)",
               f"Whole-vector softmax latency ({GPU_NAME})",
               "results/time_vs_n.png", logy=True)
    plot_sweep(rows, "gbps", "effective GB/s (12 B/element ÷ time)",
               f"Whole-vector softmax throughput ({GPU_NAME})",
               "results/bandwidth_vs_n.png", logy=False, peak=PEAK_GBPS)
    plot_scaling(srows, "results/scaling_blocks.png")

    print(f"\nCorrectness at N = {CHECK_N:,} (max relative error vs float64):")
    for name, _ in IMPLS:
        if name in errs:
            print(f"  {name:<18} {errs[name]:.3e}")
    print("\nDone. Plots + CSVs in ./results/")


if __name__ == "__main__":
    main()
