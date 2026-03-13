// Sobel edge detection reference: |Gx| + |Gy| (L1 norm)
#include <math.h>

void sobel_ref(const float* input, float* out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        int ra = (y - 1) * width;
        int rc = y * width;
        int rb = (y + 1) * width;
        for (int x = 1; x < width - 1; x++) {
            float r0a = input[ra + x - 1], r0b = input[ra + x], r0c = input[ra + x + 1];
            float r1a = input[rc + x - 1],                       r1c = input[rc + x + 1];
            float r2a = input[rb + x - 1], r2b = input[rb + x], r2c = input[rb + x + 1];
            float gx = (r0c - r0a) + 2.0f * (r1c - r1a) + (r2c - r2a);
            float gy = (r2a - r0a) + 2.0f * (r2b - r0b) + (r2c - r0c);
            out[rc + x] = fabsf(gx) + fabsf(gy);
        }
    }
}
