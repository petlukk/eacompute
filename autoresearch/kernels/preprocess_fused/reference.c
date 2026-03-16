void preprocess_fused_ref(const float *input, float *out, int len,
                         float scale, float mean, float inv_std) {
    for (int i = 0; i < len; i++) {
        float norm = input[i] * scale;
        float centered = norm - mean;
        float scaled = centered * inv_std;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 1.0f) scaled = 1.0f;
        out[i] = scaled;
    }
}
