void threshold_u8x16_ref(const unsigned char *src, unsigned char *dst, int n, unsigned char thresh) {
    // Ea's .> on u8x16 lowers to x86 pcmpgtb which is a signed byte comparison.
    // Match that semantics: cast to signed char before comparing.
    for (int i = 0; i < n; i++) {
        dst[i] = (signed char)src[i] > (signed char)thresh ? 255 : 0;
    }
}
