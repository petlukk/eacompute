# Beating NumPy's BLAS at Constant-Q Transform with 80 Lines of Eä

A SIMD kernel compiler, a DFT nobody asked for, and an honest benchmark.

---

## The Problem

Audio spectrum visualizers typically use FFT. FFT gives you linearly-spaced frequency bins — great for math, terrible for music. Human pitch perception is logarithmic: the distance from C2 to C3 (one octave) matters as much as C5 to C6. But FFT allocates equal resolution across the entire spectrum, wasting detail in the bass and over-resolving the treble.

The Constant-Q Transform (CQT) solves this. Each frequency bin gets its own window length — long windows for low frequencies (more cycles needed to resolve pitch), short windows for high frequencies. The result: 84 bins spanning 7 octaves, one per semitone, C2 through B8.

The catch: FFT can't do CQT. You need a DFT-per-bin, which is O(n*k) instead of O(n log n). The standard NumPy approach is to build a complex kernel matrix and let BLAS handle it with a single matrix-vector multiply.

We wanted to see if a hand-written SIMD kernel in [Eä](https://petlukk.github.io/eacompute/) could beat that.

## What is Eä

Eä is a compute kernel compiler. You write `.ea` files in a C-like language with explicit SIMD vector types, compile them to native shared libraries, and call them from Python with NumPy arrays. No C toolchain, no Cython, no JIT warmup.

```
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let s: f32x8 = splat(factor)
    let mut i: i32 = 0
    while i < n {
        let v: f32x8 = load(src, i)
        store(dst, i, v .* s)
        i = i + 8
    }
}
```

Eä doesn't have sin or cos intrinsics. This is a deliberate design choice — trig is a policy decision (how many polynomial terms? what accuracy? what range?) that the language refuses to hide behind a simple-looking function call. If you need trig, you precompute tables in Python or write an explicit FMA polynomial chain.

This turns out to be the key insight for CQT.

## The Design

A CQT with 84 bins (7 octaves, 12 semitones each) starting at C2 (65.4 Hz) at 44.1 kHz sample rate has these properties:

| Bin | Note | Frequency | Window Length |
|-----|------|-----------|--------------|
| 0   | C2   | 65.4 Hz   | 11,339 samples |
| 33  | A4   | 440 Hz    | 1,684 samples |
| 83  | B8   | 7,903 Hz  | 94 samples |

The quality factor Q = 16.8 (constant across all bins — that's what "constant-Q" means). Total work per frame: 200,476 FMA operations across all bins.

The approach: precompute cos/sin twiddle factor tables in Python (one-time cost, 1.6 MB), then let Eä handle the per-frame FMA loop. We bake the Hann window directly into the twiddle factors during precomputation:

```python
# Python: precompute once
for k in range(n_bins):
    n_k = int(lengths[k])
    i = np.arange(n_k, dtype=np.float64)
    window = 0.5 * (1 - np.cos(2 * np.pi * i / n_k))
    angle = 2 * np.pi * freqs[k] * i / SAMPLE_RATE
    cos_parts.append((window * np.cos(angle)).astype(np.float32))
    sin_parts.append((window * -np.sin(angle)).astype(np.float32))
```

This means the Eä kernel needs zero trig and zero windowing logic. The inner loop is pure FMA.

## The Kernel

### Version 1: Direct DFT with Dual Accumulators

The first kernel processes one frequency bin at a time. For each bin, it reads the audio segment, multiplies against precomputed cos and sin twiddle factors, accumulates, and computes the magnitude. Smoothing (exponential decay for the "falling bars" effect) is fused into the output — no separate kernel call, no intermediate array.

Two independent FMA accumulator chains (r0/r1, i0/i1) hide pipeline latency on superscalar CPUs. This is the same technique used in high-performance reduction kernels.

```
export func cqt_fused(
    audio: *f32,
    cos_table: *f32,
    sin_table: *f32,
    offsets: *i32,      // per-bin start offset into twiddle tables
    lengths: *i32,      // per-bin window length
    prev_mags: *f32,
    out_mags: *mut f32,
    alpha: f32,         // decay factor
    n_bins: i32,
    max_window_len: i32
) {
    let mut k: i32 = 0
    while k < n_bins {
        let off: i32 = offsets[k]
        let win_len: i32 = lengths[k]
        let audio_start: i32 = max_window_len - win_len

        let mut r0: f32x8 = splat(0.0)
        let mut r1: f32x8 = splat(0.0)
        let mut i0: f32x8 = splat(0.0)
        let mut i1: f32x8 = splat(0.0)

        let mut i: i32 = 0
        while i + 16 <= win_len {
            prefetch(cos_table, off + i + 64)
            prefetch(sin_table, off + i + 64)

            let s0: f32x8 = load(audio, audio_start + i)
            let s1: f32x8 = load(audio, audio_start + i + 8)
            let c0: f32x8 = load(cos_table, off + i)
            let c1: f32x8 = load(cos_table, off + i + 8)
            let sn0: f32x8 = load(sin_table, off + i)
            let sn1: f32x8 = load(sin_table, off + i + 8)

            r0 = fma(s0, c0, r0)
            r1 = fma(s1, c1, r1)
            i0 = fma(s0, sn0, i0)
            i1 = fma(s1, sn1, i1)

            i = i + 16
        }

        // 8-element and scalar tails omitted for brevity

        let real: f32 = reduce_add(r0 .+ r1)
        let imag: f32 = reduce_add(i0 .+ i1)
        let mag: f32 = sqrt(real * real + imag * imag)

        // Fused smooth decay
        let decayed: f32 = prev_mags[k] * alpha
        if mag > decayed {
            out_mags[k] = mag
        } else {
            out_mags[k] = decayed
        }

        k = k + 1
    }
}
```

The generated assembly for the hot loop is clean — the Eä compiler folds twiddle loads directly into `vfmadd231ps` memory operands, so the audio loads (`vmovups`) are the only explicit loads:

```asm
.LBB0_4:
    prefetcht0  (%rsi,%r12,4)      ; prefetch cos_table
    prefetcht0  (%rdx,%r12,4)      ; prefetch sin_table
    vmovups     (%rdi,%r12,4), %ymm5   ; load audio[i..i+7]
    vmovups     (%rdi,%r12,4), %ymm6   ; load audio[i+8..i+15]
    vfmadd231ps (%rsi,%r12,4), %ymm5, %ymm3  ; r0 += audio * cos
    vfmadd231ps (%rsi,%rbp,4), %ymm6, %ymm4  ; r1 += audio * cos
    vfmadd231ps (%rdx,%r12,4), %ymm5, %ymm2  ; i0 += audio * sin
    vfmadd231ps (%rdx,%rbp,4), %ymm6, %ymm1  ; i1 += audio * sin
```

Four FMAs per iteration, each operating on 8 floats. 64 floating-point multiply-adds per loop cycle.

### First Benchmark: The Lie

Our first benchmark compared this against a Python for-loop CQT:

```python
def numpy_cqt(audio, freqs, max_window_len):
    magnitudes = np.empty(N_BINS, dtype=np.float32)
    for k in range(N_BINS):
        n_k = int(np.ceil(Q * SAMPLE_RATE / freqs[k]))
        segment = audio[start:start + n_k]
        window = 0.5 * (1 - np.cos(2 * np.pi * i / n_k))
        kernel = window * np.exp(-2j * np.pi * freqs[k] * i / SAMPLE_RATE)
        magnitudes[k] = np.abs(np.sum(segment * kernel))
    return magnitudes
```

Result: **153x faster**. We almost shipped this number.

The problem: this "NumPy CQT" is a Python for-loop with per-bin allocations. Nobody would write production CQT this way. The honest NumPy approach is to precompute a `(84, 11339)` complex kernel matrix and do a single `np.dot`:

```python
# Precompute once: (n_bins, max_window_len) complex64 matrix
kernel_matrix = np.zeros((N_BINS, max_window_len), dtype=np.complex64)
for k in range(N_BINS):
    # ... fill in windowed complex exponentials, zero-pad shorter bins

# Per frame: single BLAS call
magnitudes = np.abs(kernel_matrix @ audio)
```

This calls directly into OpenBLAS/MKL — decades of hand-tuned assembly for matrix-vector multiply.

### Honest Benchmark v1

Methodology:
- All competitors precompute everything they can (tables, windows, indices)
- Only per-frame work is timed: transform + magnitude + smoothing
- Same smoothing (`max(current, previous * alpha)`) applied to all

With N=500 iterations and inadequate warmup:

| | Time |
|---|---|
| Eä CQT | 0.098 ms |
| NumPy matmul | 0.085 ms |

NumPy was winning. The 153x headline collapsed to **Eä being 1.1x slower**.

### Fixing the Benchmark

The N=500 result was noisy. WSL2 scheduling and insufficient warmup made the numbers unreliable. We moved to a min-of-trials methodology: 10 independent trials of 2,000 iterations each, reporting both min (best-case, no OS interference) and median (real-world scheduling).

With proper warmup (200+ iterations before timing):

| | Min | Median |
|---|---|---|
| Eä CQT | 50 us | 55 us |
| NumPy matmul | 95 us | 190 us |

Eä was already **1.9x faster at best**, **3.5x at median**. The initial "1.1x slower" was just noise.

## Optimization Attempts

### Quad Accumulators

Theory: 4 independent FMA chains (8 accumulator registers) should hide more pipeline latency than 2 chains.

Result: **slower** (100 us vs 83 us). With 8 YMM accumulator registers + 12 data registers for loads, we exceed x86-64's 16 YMM register limit. The compiler spills to stack memory, killing the very latency hiding we wanted.

### Interleaved Loads

Theory: load cos, immediately FMA, then load sin and FMA. Give the memory subsystem more time to fetch by interleaving loads with compute.

Result: **slower** (191 us vs 53 us). The original pattern — batch all loads, then batch all FMAs — is better for out-of-order execution. The CPU's reorder buffer can schedule loads far ahead of their consumers when they're grouped together.

### Merged Twiddle Table

Theory: interleave cos and sin data for each bin into a single contiguous memory region `[cos_bin0 | sin_bin0 | cos_bin1 | ...]` to halve cache line traffic.

Result: **no change** (53.8 us vs 53.1 us). The hardware prefetcher already handles two sequential streams efficiently. The second stream (sin_table) gets prefetched in parallel with the first (cos_table) because the access pattern is identical — sequential scan with the same stride.

### In-Place Smoothing

Instead of writing to `out_mags` and copying back to `prev_mags` in Python, the kernel reads and writes the same buffer:

```
export func cqt_inplace(
    audio: *f32,
    cos_table: *f32,
    sin_table: *f32,
    offsets: *i32,
    lengths: *i32,
    mags: *mut f32,    // read previous, write smoothed result
    alpha: f32,
    n_bins: i32,
    max_window_len: i32
)
```

Result: **~5% faster** (47.8 us vs 50.2 us). Modest win from eliminating a 336-byte numpy copy per frame plus one Python function call.

## Final Numbers

Methodology: 10 trials of 2,000 iterations each, 200-iteration warmup per trial, min-of-trials reported. All precomputation excluded. All competitors include smoothing. NumPy uses pre-allocated output buffers (`out=` parameter) to avoid allocation overhead.

| | Min | Median |
|---|---|---|
| **Eä CQT (fused SIMD)** | **~50 us** | **~55 us** |
| NumPy CQT (BLAS matmul) | ~95 us | ~190 us |
| NumPy FFT (wrong result) | ~106 us | ~118 us |

**Eä vs BLAS: 1.9x faster (min), 3.5x (median)**
**Eä vs FFT: 2.1x faster — and FFT gives the wrong answer**
**Throughput: 7.8 GFLOP/s**

### Why Eä Wins

**1. Eä reads 4.8x less data.**
The BLAS matmul uses a dense (84, 11339) complex64 matrix — 7.4 MB per frame, including all the zero-padded regions where shorter bins have no data. Eä's variable-length inner loops skip zeros entirely, touching only 200,476 real elements (1.6 MB).

**2. Real vs complex arithmetic.**
BLAS operates on complex64 (pairs of float32). Every BLAS FMA processes a complex multiply-add — 4 real multiplies and 2 real adds per element. Eä works directly in float32, doing 2 real FMAs per element (one for the cos component, one for sin).

**3. Fused pipeline.**
Eä does window + transform + magnitude + smoothing in one function call. NumPy requires 4 separate calls: `matmul`, `abs`, `maximum`, `copyto`. Each call crosses the Python/C boundary, allocates or writes to buffers, and makes a separate pass over the output data.

**4. Lower scheduling jitter.**
One ctypes call has less interrupt surface than four NumPy calls. This explains the median gap (3.5x) being larger than the min gap (1.9x) — Eä's single kernel call is less likely to be interrupted mid-computation.

### Why Eä Doesn't Win More

The dual-accumulator inner loop is already close to optimal for AVX2. The generated assembly has 4 `vfmadd231ps` instructions per iteration with memory operands — the compiler is folding loads into FMAs. There's no instruction waste.

The bottleneck is memory bandwidth. At 50 us for 1.6 MB of twiddle table reads, we're pulling ~32 GB/s — respectable for L3 cache, but below the theoretical memory bandwidth. The per-bin overhead (loading offsets/lengths, setting up accumulators, horizontal reduction) fragments what could be one continuous stream into 84 separate sweeps.

BLAS doesn't have this problem. Its matmul is one uninterrupted sweep through a dense matrix, which modern CPUs and prefetchers are specifically optimized for. BLAS trades compute (touching zeros) for access pattern regularity — and on some runs, when the OS cooperates perfectly, that trade nearly pays off.

## The Code

Three files. The full kernel (`cqt.ea`, 80 lines), the Python visualizer, and the benchmark. The kernel uses the language as intended: explicit types, explicit vector widths, explicit memory access. No hidden costs.

```python
import ea
import numpy as np

kernel = ea.load("cqt.ea")

# Precompute tables once
freqs = F_MIN * 2 ** (np.arange(84) / 12)
lengths = np.ceil(Q * SAMPLE_RATE / freqs).astype(np.int32)

# ... build cos/sin twiddle tables with baked-in Hann window ...

# Per frame: one call, ~50 us
kernel.cqt_fused(
    audio_buffer, cos_table, sin_table,
    offsets, lengths, prev_mags, out_mags,
    decay_alpha, n_bins, max_window_len,
)
```

The full source is on [GitHub](https://github.com/petlukk/eacompute).

## What We Learned

The first benchmark said 153x. That was a lie — comparing optimized SIMD against a Python for-loop. The second benchmark said 1.1x slower. That was noise — insufficient warmup on a noisy WSL2 scheduler. The real number is **1.9x faster**, earned by reading less data, using simpler arithmetic, and fusing the pipeline.

Three optimization attempts failed (quad accumulators, interleaved loads, merged tables) and one gave a modest 5% win (in-place smoothing). The lesson: when the compiler is already generating clean assembly, micro-optimizations rarely help. The wins came from the algorithm design — variable-length windows, baked-in windowing, fused smoothing — not from squeezing the inner loop.

Eä didn't win by being a faster compiler. It won by making it easy to write the right algorithm. A kernel that skips zeros, fuses four operations, and makes one function call will beat a kernel that touches every element of a zero-padded matrix and makes four function calls — even when the latter is backed by BLAS.
