// C reference: in-place sort using qsort-compatible interface
#include <stdlib.h>

static int cmp_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

void sort_f32_ref(float *data, int len) {
    qsort(data, (size_t)len, sizeof(float), cmp_float);
}
