// Hand-optimized C reference: clamp with AVX2/SSE intrinsics
// result[i] = clamp(data[i], lo, hi)
#include <immintrin.h>
#include <stdint.h>

// AVX2 f32x8 clamp
void clamp_f32x8_c(const float* data, float* result, float lo, float hi, int32_t len) {
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    int32_t i = 0;

    for (; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(&result[i], v);
    }

    for (; i < len; i++) {
        float v = data[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        result[i] = v;
    }
}

// SSE f32x4 clamp
void clamp_f32x4_c(const float* data, float* result, float lo, float hi, int32_t len) {
    __m128 vlo = _mm_set1_ps(lo);
    __m128 vhi = _mm_set1_ps(hi);
    int32_t i = 0;

    for (; i + 4 <= len; i += 4) {
        __m128 v = _mm_loadu_ps(&data[i]);
        v = _mm_max_ps(v, vlo);
        v = _mm_min_ps(v, vhi);
        _mm_storeu_ps(&result[i], v);
    }

    for (; i < len; i++) {
        float v = data[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        result[i] = v;
    }
}
