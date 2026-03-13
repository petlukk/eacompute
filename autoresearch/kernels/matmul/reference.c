// Scalar C reference — same triple-loop algorithm as the Eä baseline.
// Compiled with -O3 -march=native so GCC's auto-vectorizer can work.
#include <stdint.h>

void matmul_f32_c(const float* restrict a, const float* restrict b, float* restrict c, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < n; k++) {
                sum += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] = sum;
        }
    }
}
