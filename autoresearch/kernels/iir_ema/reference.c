void ema_filter_ref(const float *input, float *output, int len, float alpha) {
    if (len <= 0) return;
    float beta = 1.0f - alpha;
    output[0] = alpha * input[0];
    for (int i = 1; i < len; i++) {
        output[i] = alpha * input[i] + beta * output[i - 1];
    }
}
