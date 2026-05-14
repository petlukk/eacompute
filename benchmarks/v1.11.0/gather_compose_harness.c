/* gather_compose_harness.c
 *
 * LUT gather benchmark for v1.11.0 Phase 6 perf audit.
 *
 * Three kernels (x86) / two kernels (ARM):
 *   scalar_loop  — pure scalar loads & stores. Cross-platform baseline.
 *   compose_x4   — scalar loads + f32x4_from_scalars + vector store.
 *   gather_x86   — AVX2 vgatherdps. **x86-only.**
 *
 * Build for x86:
 *   ../../target/release/ea gather_compose_bench.ea --lib
 *   cc gather_compose_harness.c -L. -l:gather_compose_bench.so -lm -O2 \
 *       -Wl,-rpath,'$ORIGIN' -DBUILD_X86 -o gather_compose_bench
 *   taskset -c 0 ./gather_compose_bench
 *
 * Build for ARM (cross from x86):
 *   ../../target/release/ea gather_compose_bench_arm.ea \
 *       --target-triple=aarch64-unknown-linux-gnu
 *   aarch64-linux-gnu-gcc -shared -o libgather_compose_bench_arm.so \
 *       gather_compose_bench_arm.o
 *   aarch64-linux-gnu-gcc gather_compose_harness.c \
 *       -L. -lgather_compose_bench_arm -lm -O2 \
 *       -Wl,-rpath,'$ORIGIN' \
 *       -o gather_compose_bench_arm
 *   scp libgather_compose_bench_arm.so gather_compose_bench_arm pi:~/
 *   ssh pi 'taskset -c 0 ~/gather_compose_bench_arm'
 *
 * Workload: 16K-entry LUT (64 KB), 64K outputs in groups of 4 indices.
 * LUT is mostly L1d-resident; indices are sequential-ish so we don't
 * trigger pathological cache eviction. This is the IQ3 LUT dequant
 * shape.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

extern void scalar_loop(const float *lut, const int32_t *indices, float *out, int n);
extern void compose_x4(const float *lut, const int32_t *indices, float *out, int n);
#ifdef BUILD_X86
extern void gather_x86(const float *lut, const int32_t *indices, float *out, int n);
#endif

#define LUT_SIZE 16384
#define N        65536
#define N_INNER  500
#define N_RUNS   10
#define N_WARM   10

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

static void absorb(const float *p, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += p[i];
    g_sink = acc;
}

static int verify(const float *lut, const int32_t *indices, int n) {
    static float a[N], b[N];
    scalar_loop(lut, indices, a, n);
    compose_x4(lut, indices, b, n);
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            fprintf(stderr, "VERIFY FAIL compose: i=%d scalar=%g compose=%g\n",
                    i, a[i], b[i]);
            return 0;
        }
    }
#ifdef BUILD_X86
    static float c[N];
    gather_x86(lut, indices, c, n);
    for (int i = 0; i < n; ++i) {
        if (a[i] != c[i]) {
            fprintf(stderr, "VERIFY FAIL gather: i=%d scalar=%g gather=%g\n",
                    i, a[i], c[i]);
            return 0;
        }
    }
#endif
    fprintf(stderr, "verify OK: all three kernels produce identical output\n");
    return 1;
}

static double bench(void (*k)(const float *, const int32_t *, float *, int),
                    const float *lut, const int32_t *indices, float *out, int n) {
    double t0 = now_ns();
    for (int j = 0; j < N_INNER; ++j) {
        k(lut, indices, out, n);
        absorb(out, n);
    }
    return (now_ns() - t0) / (double)N_INNER;
}

int main(void) {
    static float lut[LUT_SIZE];
    static int32_t indices[N];
    static float out[N];

    /* LUT: simple linear ramp. */
    for (int i = 0; i < LUT_SIZE; ++i) lut[i] = (float)i * 0.001f;

    /* Indices: deterministic LCG modulo LUT_SIZE, no out-of-bounds. */
    unsigned int state = 0xBADC0FEEu;
    for (int i = 0; i < N; ++i) {
        state = state * 1664525u + 1013904223u;
        indices[i] = (int32_t)(state % LUT_SIZE);
    }

    if (!verify(lut, indices, N)) return 1;

    /* Warm-up. */
    for (int k = 0; k < N_WARM; ++k) {
        scalar_loop(lut, indices, out, N); absorb(out, N);
        compose_x4(lut, indices, out, N);  absorb(out, N);
#ifdef BUILD_X86
        gather_x86(lut, indices, out, N);  absorb(out, N);
#endif
    }

    double sc_ns[N_RUNS], cp_ns[N_RUNS];
#ifdef BUILD_X86
    double gt_ns[N_RUNS];
#endif
    for (int r = 0; r < N_RUNS; ++r) {
        sc_ns[r] = bench(scalar_loop, lut, indices, out, N);
        cp_ns[r] = bench(compose_x4,  lut, indices, out, N);
#ifdef BUILD_X86
        gt_ns[r] = bench(gather_x86,  lut, indices, out, N);
#endif
    }
    qsort(sc_ns, N_RUNS, sizeof(double), cmp_double);
    qsort(cp_ns, N_RUNS, sizeof(double), cmp_double);
    double sc_med = sc_ns[N_RUNS / 2];
    double cp_med = cp_ns[N_RUNS / 2];

#ifdef BUILD_X86
    qsort(gt_ns, N_RUNS, sizeof(double), cmp_double);
    double gt_med = gt_ns[N_RUNS / 2];
    fprintf(stderr, "gather_compose v1.11.0 — x86 — n=%d outputs, %d-entry LUT, median of %d runs of %d calls\n",
            N, LUT_SIZE, N_RUNS, N_INNER);
    fprintf(stderr, "  scalar_loop:   %8.2f us/call\n", sc_med / 1000.0);
    fprintf(stderr, "  compose_x4:    %8.2f us/call  (compose / scalar = %.2fx)\n",
            cp_med / 1000.0, sc_med / cp_med);
    fprintf(stderr, "  gather_x86:    %8.2f us/call  (gather / scalar = %.2fx)\n",
            gt_med / 1000.0, sc_med / gt_med);
    fprintf(stderr, "  sink: %g\n", (double)g_sink);

    printf("{\"kernel\":\"scalar_loop\",\"median_ns\":%lu,\"n_inner\":%d,\"n_runs\":%d}\n",
           (unsigned long)sc_med, N_INNER, N_RUNS);
    printf("{\"kernel\":\"compose_x4\",\"median_ns\":%lu,\"n_inner\":%d,\"n_runs\":%d}\n",
           (unsigned long)cp_med, N_INNER, N_RUNS);
    printf("{\"kernel\":\"gather_x86\",\"median_ns\":%lu,\"n_inner\":%d,\"n_runs\":%d}\n",
           (unsigned long)gt_med, N_INNER, N_RUNS);
#else
    fprintf(stderr, "gather_compose v1.11.0 — ARM — n=%d outputs, %d-entry LUT, median of %d runs of %d calls\n",
            N, LUT_SIZE, N_RUNS, N_INNER);
    fprintf(stderr, "  scalar_loop:   %8.2f us/call\n", sc_med / 1000.0);
    fprintf(stderr, "  compose_x4:    %8.2f us/call  (compose / scalar = %.2fx — spec target: >= 1.0x)\n",
            cp_med / 1000.0, sc_med / cp_med);
    fprintf(stderr, "  sink: %g\n", (double)g_sink);

    printf("{\"kernel\":\"scalar_loop\",\"median_ns\":%lu,\"n_inner\":%d,\"n_runs\":%d}\n",
           (unsigned long)sc_med, N_INNER, N_RUNS);
    printf("{\"kernel\":\"compose_x4\",\"median_ns\":%lu,\"n_inner\":%d,\"n_runs\":%d}\n",
           (unsigned long)cp_med, N_INNER, N_RUNS);
#endif
    return 0;
}
