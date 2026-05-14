/* fp16_kv_harness.c
 *
 * Pi 5 FP16 KV-path benchmark harness for v1.11.0 Phase 6 perf audit.
 * Compares kv_roundtrip (f16 storage / f32 compute / cvt per load) against
 * kv_native (f16 storage / f16 compute) from fp16_kv_bench.ea.
 *
 * REQUIRES: Cortex-A76 with FEAT_FP16 (Pi 5). Cannot run on x86 — no
 * native f16x8 hardware. Cross-compile + scp + run on Pi 5.
 *
 * Cross-compile recipe (assumes you have aarch64-linux-gnu-gcc installed):
 *
 *   # 1. Build the kernel for ARM:
 *   ../../target/release/ea fp16_kv_bench.ea \
 *       --fp16 --target-triple=aarch64-unknown-linux-gnu
 *   # (produces fp16_kv_bench.o — ARM aarch64 ELF)
 *
 *   # 2. Link a shared lib for ARM using the cross-linker:
 *   aarch64-linux-gnu-gcc -shared -o libfp16_kv_bench.so fp16_kv_bench.o -lm
 *
 *   # 3. Cross-compile the harness:
 *   aarch64-linux-gnu-gcc fp16_kv_harness.c -L. -lfp16_kv_bench -lm \
 *       -march=armv8-a+fp16 -O2 -Wl,-rpath,'$ORIGIN' \
 *       -o fp16_kv_bench
 *
 *   # 4. Ship to Pi:
 *   scp libfp16_kv_bench.so fp16_kv_bench pi:~/
 *   ssh pi 'taskset -c 0 ~/fp16_kv_bench'
 *
 * Methodology:
 *   - n = 4096 (close to a real KV-cache row width, fits in L1d).
 *   - Random f16 values in [-1, 1].
 *   - 10 warm-up calls, then median of 10 runs of 1000 inner calls.
 *   - Compiler can't elide the loop: output is read back into a volatile
 *     sink.
 *
 * Verify pass: outputs of the two kernels should match within ~5% rel
 * because f16 rounding rules differ between the cvt-and-back path and
 * the all-f16 path. Within tolerance we accept; loud failure is FAIL.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

/* Declarations match the ARM-built fp16_kv_bench.so.
 * Use _Float16 (gcc / clang on aarch64 with -march=armv8-a+fp16). */
extern void kv_roundtrip(const int16_t *x, float scale, int16_t *out, int n);
extern void kv_native(const _Float16 *x, _Float16 scale, _Float16 *out, int n);

#define N 4096
#define N_INNER 1000
#define N_RUNS  10
#define N_WARM  10

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static volatile float g_sink = 0.0f;

static void absorb_i16(const int16_t *p, int n) {
    int32_t acc = 0;
    for (int i = 0; i < n; ++i) acc += p[i];
    g_sink = (float)acc;
}

static void absorb_f16(const _Float16 *p, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += (float)p[i];
    g_sink = acc;
}

static int verify(const _Float16 *xf16, const int16_t *xi16, int n) {
    static int16_t a[N];
    static _Float16 b[N];
    kv_roundtrip(xi16, 1.0f, a, n);
    kv_native(xf16, (_Float16)1.0f, b, n);

    /* The bit-pattern of int16_t a[i] is the f16 representation. Reinterpret. */
    const _Float16 *af16 = (const _Float16 *)a;
    float max_rel = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < n; ++i) {
        float ya = (float)af16[i];
        float yb = (float)b[i];
        float ref = fabsf(ya) > 1e-6f ? fabsf(ya) : 1e-6f;
        float rel = fabsf(ya - yb) / ref;
        if (rel > max_rel) { max_rel = rel; max_idx = i; }
    }
    /* The cvt-and-back path rounds the intermediate f32 product to f16
     * after the multiply; the native path keeps everything in f16 the
     * whole way, which has slightly different rounding semantics for the
     * sumsq (f32 vs f16 accumulator). 5% rel is generous but not absurd
     * for a sumsq that compounds n=4096 contributions in f16. */
    if (max_rel > 0.05f) {
        fprintf(stderr, "VERIFY FAIL: max rel %g at i=%d roundtrip=%g native=%g\n",
                max_rel, max_idx, (double)af16[max_idx], (double)b[max_idx]);
        return 0;
    }
    fprintf(stderr, "verify OK: max rel %g (< 5%% tolerance for f32-vs-f16 accumulator)\n",
            max_rel);
    return 1;
}

static double bench_roundtrip(const int16_t *x, int16_t *out, int n) {
    double t0 = now_ns();
    for (int k = 0; k < N_INNER; ++k) {
        kv_roundtrip(x, 1.5f, out, n);
        absorb_i16(out, n);
    }
    return (now_ns() - t0) / (double)N_INNER;
}

static double bench_native(const _Float16 *x, _Float16 *out, int n) {
    double t0 = now_ns();
    for (int k = 0; k < N_INNER; ++k) {
        kv_native(x, (_Float16)1.5f, out, n);
        absorb_f16(out, n);
    }
    return (now_ns() - t0) / (double)N_INNER;
}

int main(void) {
    static _Float16 x_f16[N];
    static int16_t  x_i16[N];
    static _Float16 out_f16[N];
    static int16_t  out_i16[N];

    /* Deterministic LCG fill in [-1, 1] f16 bits. */
    unsigned int state = 0xCAFEF00Du;
    for (int i = 0; i < N; ++i) {
        state = state * 1664525u + 1013904223u;
        float u = (float)state / (float)0xFFFFFFFFu;
        float f = u * 2.0f - 1.0f;
        _Float16 h = (_Float16)f;
        x_f16[i] = h;
        /* Bit-aliased i16 view of same f16 storage. */
        memcpy(&x_i16[i], &h, sizeof(int16_t));
    }

    if (!verify(x_f16, x_i16, N)) return 1;

    for (int k = 0; k < N_WARM; ++k) {
        kv_roundtrip(x_i16, 1.5f, out_i16, N); absorb_i16(out_i16, N);
        kv_native(x_f16, (_Float16)1.5f, out_f16, N); absorb_f16(out_f16, N);
    }

    double rt_ns[N_RUNS], nv_ns[N_RUNS];
    for (int r = 0; r < N_RUNS; ++r) {
        rt_ns[r] = bench_roundtrip(x_i16, out_i16, N);
        nv_ns[r] = bench_native(x_f16, out_f16, N);
    }
    qsort(rt_ns, N_RUNS, sizeof(double), cmp_double);
    qsort(nv_ns, N_RUNS, sizeof(double), cmp_double);

    double rt_med = rt_ns[N_RUNS / 2];
    double nv_med = nv_ns[N_RUNS / 2];

    fprintf(stderr, "fp16_kv v1.11.0 — Pi 5 Cortex-A76 + FEAT_FP16 — n=%d, median of %d runs of %d calls\n",
            N, N_RUNS, N_INNER);
    fprintf(stderr, "  kv_roundtrip (f32 compute, cvt_f16_f32 per load): %8.3f us/call\n", rt_med / 1000.0);
    fprintf(stderr, "  kv_native    (f16 compute end-to-end):            %8.3f us/call\n", nv_med / 1000.0);
    fprintf(stderr, "  speedup:                                          %.2fx\n", rt_med / nv_med);
    fprintf(stderr, "  sink: %g\n", (double)g_sink);

    printf("{\"kernel\":\"kv_roundtrip\",\"median_ns\":%lu,\"n_inner\":%d,\"n_runs\":%d}\n",
           (unsigned long)rt_med, N_INNER, N_RUNS);
    printf("{\"kernel\":\"kv_native\",\"median_ns\":%lu,\"n_inner\":%d,\"n_runs\":%d}\n",
           (unsigned long)nv_med, N_INNER, N_RUNS);
    return 0;
}
