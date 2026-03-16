#include <stdint.h>

static uint8_t classify_byte(uint8_t b) {
    if (b == 32 || b == 9 || b == 10 || b == 13) return 1;
    if ((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) return 2;
    if (b >= 48 && b <= 57) return 4;
    if (b >= 33 && b <= 126) return 8;
    return 0;
}

void text_prepass_fused_ref(const uint8_t *text, uint8_t *flags,
                            uint8_t *lower, uint8_t *boundaries, int len) {
    if (len <= 0) return;

    uint8_t f0 = classify_byte(text[0]);
    flags[0] = f0;
    lower[0] = (text[0] >= 65 && text[0] <= 90) ? text[0] + 32 : text[0];
    boundaries[0] = 1;

    for (int i = 1; i < len; i++) {
        uint8_t b = text[i];
        uint8_t f = classify_byte(b);
        uint8_t fp = classify_byte(text[i - 1]);

        flags[i] = f;
        lower[i] = (b >= 65 && b <= 90) ? b + 32 : b;
        boundaries[i] = (f != fp) ? 1 : 0;
    }
}
