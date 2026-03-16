#include <math.h>

void brightness_contrast_ref(const unsigned char *input, float *output,
                              int len, float brightness, float contrast) {
    float scale = contrast / 255.0f;
    for (int i = 0; i < len; i++) {
        float val = (float)input[i] * scale + brightness;
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        output[i] = val;
    }
}
