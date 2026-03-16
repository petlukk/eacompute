#include <float.h>

void frame_stats_ref(
    const float *data,
    int len,
    float *out_min,
    float *out_max,
    float *out_sum
) {
    float s = 0.0f;
    float mn = data[0];
    float mx = data[0];
    for (int i = 0; i < len; i++) {
        float v = data[i];
        s += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    out_min[0] = mn;
    out_max[0] = mx;
    out_sum[0] = s;
}
