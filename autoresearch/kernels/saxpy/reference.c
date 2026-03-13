// Hand-optimized C reference: SAXPY with AVX2/SSE intrinsics
// y[i] = a * x[i] + y[i]
#include <immintrin.h>
#include <stdint.h>

// AVX2 f32x8 saxpy
void saxpy_f32x8_c(float a, const float* x, float* y, int32_t len) {
    __m256 va = _mm256_set1_ps(a);
    int32_t i = 0;

    for (; i + 8 <= len; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);
        vy = _mm256_fmadd_ps(va, vx, vy);
        _mm256_storeu_ps(&y[i], vy);
    }

    for (; i < len; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// SSE f32x4 saxpy
void saxpy_f32x4_c(float a, const float* x, float* y, int32_t len) {
    __m128 va = _mm_set1_ps(a);
    int32_t i = 0;

    for (; i + 4 <= len; i += 4) {
        __m128 vx = _mm_loadu_ps(&x[i]);
        __m128 vy = _mm_loadu_ps(&y[i]);
        vy = _mm_fmadd_ps(va, vx, vy);
        _mm_storeu_ps(&y[i], vy);
    }

    for (; i < len; i++) {
        y[i] = a * x[i] + y[i];
    }
}
