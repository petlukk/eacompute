void histogram_i32_c(const int* data, int* hist, int len) {
    for (int i = 0; i < 256; i++) hist[i] = 0;
    for (int i = 0; i < len; i++) hist[data[i]]++;
}
