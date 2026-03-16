#include <stdint.h>

int16_t dot_u8i8_ref(const uint8_t *act, const int8_t *wt, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (int16_t)act[i] * (int16_t)wt[i];
    }
    return (int16_t)sum;
}
