#include <stdint.h>

void lut_apply_gather_ref(const uint8_t *data, const float *lut, float *out, int len) {
    for (int i = 0; i < len; i++) {
        out[i] = lut[data[i]];
    }
}
