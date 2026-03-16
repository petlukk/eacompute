void scatter_add_ref(const float *values, const int *indices, const float *mask,
                     float *output, int len, float threshold) {
    for (int i = 0; i < len; i++) {
        if (mask[i] > threshold) {
            int idx = indices[i];
            output[idx] = output[idx] + values[i];
        }
    }
}
