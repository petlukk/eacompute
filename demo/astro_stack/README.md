# Astronomy Frame Stacking — Eä Demo

Stacks N noisy exposures to reduce noise using real telescope data
from NASA SkyView (M31 / Andromeda galaxy). Demonstrates that
**iteration fusion** — batching multiple frames per pass over the
accumulator — beats NumPy's SIMD-optimized ufuncs by 1.25–1.76x on a
workload previously considered bandwidth-bound.

## Key result

4096×4096, 16 frames, single-threaded, 30 runs, median:

```
NumPy streaming (np.add loop)  : 119.3 ms
Eä single-frame (same loop)    : 109.7 ms   ← matches NumPy (bandwidth-bound)
Eä batched (8 frames/pass)     :  67.8 ms   ← 1.76x faster
```

The single-frame kernel hits the same memory bandwidth wall as NumPy.
The batched kernel breaks through it by fusing iterations. Speedup
increases with image size as acc memory traffic moves from cache to DRAM.

## Why batching wins

Single-frame accumulation `acc[i] += frame[i]` makes N passes over the
accumulator. Each pass reads and writes acc — 3 memory transactions per
element per frame. With N=16 frames, that's 48 transactions per element.

Batched accumulation processes K frames per pass:
```
acc[i] += f0[i] + f1[i] + f2[i] + ... + f7[i]   // K=8 frames, one pass
```

This reduces acc traffic from `3N` to `3⌈N/K⌉` transactions per element.
For N=16, K=8: from 48 down to 6 — a 2.7x traffic reduction.

The gap between traffic reduction (2.7x) and wall-clock speedup is
expected: frame loads dominate at high batch sizes. With K=8, each pass
reads 8 frames plus one acc read/write. The acc savings are large in
proportion but frame loads are the majority of total traffic. As image
size grows (and acc moves to slower DRAM), the speedup increases from
1.25x at 512² to 1.76x at 4096².

## Scale sweep

Bandwidth analysis across image sizes, calibrated against STREAM peak
(~21 GB/s single-threaded on this machine):

```
Size      AccMB   NP stream   Eä batch   Speedup   NP GB/s   Bat GB/s
 512²       1      0.5 ms      0.4 ms     1.25x      99.4      46.5
1024²       4      4.3 ms      2.8 ms     1.53x      47.1      27.1
2048²      16     21.8 ms     13.3 ms     1.63x      37.0      22.6
4096²      64    119.3 ms     67.8 ms     1.76x      27.0      17.8
```

Key observations:
- **Speedup increases with size**: 1.25x → 1.53x → 1.63x → 1.76x.
  As acc moves from L2/L3 to DRAM, reducing acc traffic matters more.
- **512² (L2-resident, 1MB)**: Both methods fast, batching still wins
  but acc is already cached so the savings are smaller.
- **4096² (DRAM-bound, 64MB)**: Peak 1.76x — acc round-trips to DRAM
  are expensive and batching eliminates most of them.
- **NP GB/s > STREAM peak at 512²**: Data fits in L2, so effective
  bandwidth exceeds DRAM-measured STREAM peak.

## Non-temporal stores (stream_store)

We tested `stream_store` (x86 `vmovntps`) — bypasses cache on write-back.
Theory: avoids write-allocate overhead when acc exceeds L3 cache.

Result: **stream_store never wins** on this workload. The dual-accumulator
loop writes to `i` and `i+8` (16 floats = 64 bytes = one cache line),
so regular stores already cover full cache lines with no write-allocate
waste. Stream stores just evict acc from cache between batches, hurting
performance.

## The kernels

**Single-frame accumulation** — same as NumPy, one frame per pass:
```
export func accumulate_f32x8(acc: *mut f32, frame: *restrict f32, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        prefetch(acc, i + 64)
        prefetch(frame, i + 64)
        let va0: f32x8 = load(acc, i)
        let vf0: f32x8 = load(frame, i)
        let va1: f32x8 = load(acc, i + 8)
        let vf1: f32x8 = load(frame, i + 8)
        store(acc, i, va0 .+ vf0)
        store(acc, i + 8, va1 .+ vf1)
        i = i + 16
    }
    // ... residual + scalar tail
}
```

**Batched accumulation** — 8 frames per pass, 2.7x less acc traffic:
```
export func accumulate_batch8_f32x8(
    acc: *mut f32,
    f0: *restrict f32, f1: *restrict f32, ..., f7: *restrict f32,
    len: i32
) {
    let mut i: i32 = 0
    while i + 16 <= len {
        prefetch(acc, i + 64)
        prefetch(f0, i + 64)
        // ... prefetch all 8 frames

        let a0: f32x8 = load(acc, i)
        let a1: f32x8 = load(acc, i + 8)
        // ... load all 8 frames at both offsets

        store(acc, i,     a0 .+ v00 .+ v10 .+ v20 .+ v30 .+ v40 .+ v50 .+ v60 .+ v70)
        store(acc, i + 8, a1 .+ v01 .+ v11 .+ v21 .+ v31 .+ v41 .+ v51 .+ v61 .+ v71)
        i = i + 16
    }
    // ... residual + scalar tail
}
```

Python calls the batched kernel with frame grouping:
```python
i = 0
while i + 8 <= n_frames:
    stack.accumulate_batch8_f32x8(acc, frames[i], ..., frames[i+7])
    i += 8
while i + 4 <= n_frames:
    stack.accumulate_batch4_f32x8(acc, frames[i], ..., frames[i+3])
    i += 4
while i < n_frames:
    stack.accumulate_f32x8(acc, frames[i])
    i += 1
```

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Build kernel + generate bindings
cd demo/astro_stack && bash build.sh

# Run demo (correctness + performance)
python run.py

# Scale sweep (512² to 4096², bandwidth analysis)
python bench_scale.py

# Plot scale sweep results
python plot_scale.py
```

## How it works

On first run, downloads a DSS telescope image of M31 from NASA SkyView,
normalizes it, and generates N noisy exposures. Falls back to synthetic
starfield if download fails.

```bash
ea stack.ea --lib -o libstack.so   # → libstack.so + stack.ea.json
ea bind stack.ea --python          # → stack.py (auto-generated)
```
