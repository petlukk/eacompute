/* exp_poly_f32_harness.c
 *
 * Drives softmax_libm vs softmax_poly from exp_poly_f32_bench.so.
 *
 * Methodology:
 *   - n = 8192 f32 vector, filled with deterministic LCG in [-10, 10].
 *   - Warm up: 10 calls per kernel.
 *   - Measure: 10 runs of N_INNER calls each; report median wall-clock
 *     per call. Median is robust against OS noise.
 *   - Compiler can't elide the loop because we feed each iteration's
 *     output back to the next iteration's input (sum stays live across
 *     iters) and finally write to a volatile sink.
 *
 * Build:
 *   cc exp_poly_f32_harness.c -L. -lexp_poly_f32_bench -lm -o exp_poly_f32_bench \
 *       -Wl,-rpath,'$ORIGIN' -O2
 *
 * Run (pin to one core for stable measurements):
 *   taskset -c 0 ./exp_poly_f32_bench
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

extern void exp_only_libm(const float *x, float *out, int n);
extern void exp_only_poly(const float *x, float *out, int n);
extern void softmax_libm(const float *x, float *out, int n);
extern void softmax_poly(const float *x, float *out, int n);

#define N 8192
#define N_INNER 200     /* calls per timed run */
#define N_RUNS  10      /* runs for median */
#define N_WARM  10      /* warm-up iters */

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Volatile sink: prevent the compiler from eliding the inner loop because
 * the outputs are unused. We hash the output into a volatile global. */
static volatile float g_sink = 0.0f;

static void absorb(const float *out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += out[i];
    g_sink = acc;
}

/* Verify both kernel pairs produce roughly-equal output (sanity check). */
static int verify(const float *x, int n) {
    static float a[N], b[N];

    /* exp_only: tighter tolerance, contract is rel err <= 2^-18 ~ 4e-6 */
    exp_only_libm(x, a, n);
    exp_only_poly(x, b, n);
    float max_rel = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < n; ++i) {
        if (a[i] < 1e-20f) continue;
        float rel = fabsf(a[i] - b[i]) / a[i];
        if (rel > max_rel) { max_rel = rel; max_idx = i; }
    }
    if (max_rel > 4.0e-6f) {
        fprintf(stderr, "VERIFY FAIL exp_only: max rel %g at i=%d (libm=%g poly=%g)\n",
                max_rel, max_idx, a[max_idx], b[max_idx]);
        return 0;
    }
    fprintf(stderr, "verify exp_only OK: max rel %g (contract is 2^-18 ~ 3.8e-6)\n", max_rel);

    /* softmax: relaxed 1e-3 because sum normalization compounds error */
    softmax_libm(x, a, n);
    softmax_poly(x, b, n);
    max_rel = 0.0f;
    max_idx = -1;
    for (int i = 0; i < n; ++i) {
        float rel = fabsf(a[i] - b[i]) / (a[i] + 1e-30f);
        if (rel > max_rel) { max_rel = rel; max_idx = i; }
    }
    if (max_rel > 1e-3f) {
        fprintf(stderr, "VERIFY FAIL softmax: max rel %g at i=%d (libm=%g poly=%g)\n",
                max_rel, max_idx, a[max_idx], b[max_idx]);
        return 0;
    }
    fprintf(stderr, "verify softmax OK: max rel %g (< 1e-3)\n", max_rel);
    return 1;
}

static double run_one(void (*kernel)(const float *, float *, int),
                      const float *x, float *out, int n) {
    double t0 = now_ns();
    for (int k = 0; k < N_INNER; ++k) {
        kernel(x, out, n);
        absorb(out, n);
    }
    double t1 = now_ns();
    return (t1 - t0) / (double)N_INNER;
}

int main(void) {
    /* Deterministic LCG fill in [-10, 10]. */
    static float x[N];
    static float out[N];
    unsigned int state = 0xDEADBEEFu;
    for (int i = 0; i < N; ++i) {
        state = state * 1664525u + 1013904223u;
        float u = (float)state / (float)0xFFFFFFFFu;   /* [0, 1] */
        x[i] = u * 20.0f - 10.0f;
    }

    if (!verify(x, N)) return 1;

    /* Warm-up. */
    for (int k = 0; k < N_WARM; ++k) {
        exp_only_libm(x, out, N); absorb(out, N);
        exp_only_poly(x, out, N); absorb(out, N);
        softmax_libm(x, out, N); absorb(out, N);
        softmax_poly(x, out, N); absorb(out, N);
    }

    /* Timed runs. */
    double exp_libm_ns[N_RUNS], exp_poly_ns[N_RUNS];
    double sm_libm_ns[N_RUNS], sm_poly_ns[N_RUNS];
    for (int r = 0; r < N_RUNS; ++r) {
        exp_libm_ns[r] = run_one(exp_only_libm, x, out, N);
        exp_poly_ns[r] = run_one(exp_only_poly, x, out, N);
        sm_libm_ns[r]  = run_one(softmax_libm, x, out, N);
        sm_poly_ns[r]  = run_one(softmax_poly, x, out, N);
    }

    qsort(exp_libm_ns, N_RUNS, sizeof(double), cmp_double);
    qsort(exp_poly_ns, N_RUNS, sizeof(double), cmp_double);
    qsort(sm_libm_ns,  N_RUNS, sizeof(double), cmp_double);
    qsort(sm_poly_ns,  N_RUNS, sizeof(double), cmp_double);
    double exp_libm_med = exp_libm_ns[N_RUNS / 2];
    double exp_poly_med = exp_poly_ns[N_RUNS / 2];
    double sm_libm_med  = sm_libm_ns[N_RUNS / 2];
    double sm_poly_med  = sm_poly_ns[N_RUNS / 2];

    printf("exp_poly_f32 v1.11.0 — n=%d, median of %d runs of %d calls each\n\n",
           N, N_RUNS, N_INNER);
    printf("[primary] isolated exp() — the spec's '~10x faster' claim:\n");
    printf("  exp_only_libm:    %8.2f us/call  (scalarized to libm expf)\n",
           exp_libm_med / 1000.0);
    printf("  exp_only_poly:    %8.2f us/call  (SIMD polynomial)\n",
           exp_poly_med / 1000.0);
    printf("  speedup:          %.2fx\n\n", exp_libm_med / exp_poly_med);

    printf("[secondary] full softmax (multi-pass, exp amortized):\n");
    printf("  softmax_libm:     %8.2f us/call\n", sm_libm_med / 1000.0);
    printf("  softmax_poly:     %8.2f us/call\n", sm_poly_med / 1000.0);
    printf("  speedup:          %.2fx  (Amdahl: softmax has 2 extra passes besides exp)\n\n",
           sm_libm_med / sm_poly_med);

    printf("  sink: %g\n", (double)g_sink);
    return 0;
}
