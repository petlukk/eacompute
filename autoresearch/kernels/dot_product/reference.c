// Hand-optimized C reference: dot product with AVX2/SSE intrinsics
#include <immintrin.h>
#include <stdint.h>

// AVX2 f32x8 dot product
float dot_f32x8_c(const float* a, const float* b, int32_t len) {
    __m256 acc = _mm256_setzero_ps();
    int32_t i = 0;

    for (; i + 8 <= len; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    // Horizontal sum: reduce 8 floats to 1
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    shuf = _mm_movehl_ps(sum2, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    float total = _mm_cvtss_f32(sum1);

    for (; i < len; i++) {
        total += a[i] * b[i];
    }
    return total;
}

// SSE f32x4 dot product
float dot_f32x4_c(const float* a, const float* b, int32_t len) {
    __m128 acc = _mm_setzero_ps();
    int32_t i = 0;

    for (; i + 4 <= len; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        acc = _mm_fmadd_ps(va, vb, acc);
    }

    __m128 shuf = _mm_movehdup_ps(acc);
    __m128 sum2 = _mm_add_ps(acc, shuf);
    shuf = _mm_movehl_ps(sum2, sum2);
    __m128 sum1 = _mm_add_ss(sum2, shuf);
    float total = _mm_cvtss_f32(sum1);

    for (; i < len; i++) {
        total += a[i] * b[i];
    }
    return total;
}
