// Scalar C reference implementation for inclusive prefix sum
#include <stdint.h>

void prefix_sum_f32_c(const float* data, float* out, int32_t len) {
    if (len <= 0) return;
    out[0] = data[0];
    for (int32_t i = 1; i < len; i++) {
        out[i] = out[i-1] + data[i];
    }
}
