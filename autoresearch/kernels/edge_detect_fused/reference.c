// Unfused edge detection reference: gaussian blur 3x3 -> sobel magnitude -> threshold
// Three separate passes over the image (naive, no fusion).
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void gaussian_blur_3x3_ref(const float* input, float* out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        int ra = (y - 1) * width;
        int rc = y * width;
        int rb = (y + 1) * width;
        for (int x = 1; x < width - 1; x++) {
            float r0a = input[ra + x - 1], r0b = input[ra + x], r0c = input[ra + x + 1];
            float r1a = input[rc + x - 1], r1b = input[rc + x], r1c = input[rc + x + 1];
            float r2a = input[rb + x - 1], r2b = input[rb + x], r2c = input[rb + x + 1];
            out[rc + x] = (r0a + r0c + r2a + r2c
                           + 2.0f * (r0b + r1a + r1c + r2b)
                           + 4.0f * r1b) * 0.0625f;
        }
    }
}

static void sobel_magnitude_ref(const float* input, float* out, int width, int height) {
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

static void threshold_ref(const float* input, float* out, int len, float thresh) {
    for (int i = 0; i < len; i++) {
        out[i] = input[i] > thresh ? 1.0f : 0.0f;
    }
}

void edge_detect_unfused_ref(
    const float* input, float* out, int width, int height, float thresh
) {
    int total = width * height;
    float* blur_buf = (float*)calloc(total, sizeof(float));
    float* sobel_buf = (float*)calloc(total, sizeof(float));

    gaussian_blur_3x3_ref(input, blur_buf, width, height);
    sobel_magnitude_ref(blur_buf, sobel_buf, width, height);
    threshold_ref(sobel_buf, out, total, thresh);

    free(blur_buf);
    free(sobel_buf);
}
