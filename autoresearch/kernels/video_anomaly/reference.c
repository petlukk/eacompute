#include <math.h>

float anomaly_count_fused_ref(const float *a, const float *b, int len, float thresh) {
    float total = 0.0f;
    for (int i = 0; i < len; i++) {
        float d = a[i] - b[i];
        float abs_d = d < 0.0f ? -d : d;
        if (abs_d > thresh) {
            total += 1.0f;
        }
    }
    return total;
}
