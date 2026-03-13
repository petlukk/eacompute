// 3x3 depthwise int8 convolution reference (matches pmaddubsw+pmaddwd behavior)
// src: (H+2)x(W+2)xC_in padded uint8, wt: 9xC_in int8, dst: HxW int32
#include <stdint.h>

static inline int16_t sat_i16(int32_t x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

void conv2d_3x3_ref(const uint8_t* src, const int8_t* wt, int32_t* dst,
                     int H, int W, int C_in) {
    int stride = (W + 2) * C_in;
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int32_t acc = 0;
            for (int dr = 0; dr < 3; dr++) {
                for (int dc = 0; dc < 3; dc++) {
                    int src_off = (row + dr) * stride + (col + dc) * C_in;
                    int wt_off = (dr * 3 + dc) * C_in;
                    // Match pmaddubsw+pmaddwd: pairs saturate at i16, then widen to i32
                    for (int ci = 0; ci < C_in; ci += 2) {
                        int16_t p0 = sat_i16((int32_t)src[src_off+ci] * (int32_t)wt[wt_off+ci]
                                           + (int32_t)src[src_off+ci+1] * (int32_t)wt[wt_off+ci+1]);
                        acc += (int32_t)p0;
                    }
                }
            }
            dst[row * W + col] = acc;
        }
    }
}
