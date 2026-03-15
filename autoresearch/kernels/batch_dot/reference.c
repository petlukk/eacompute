// C AVX2 reference: batch dot product
#include <immintrin.h>

void batch_dot_ref(
    const float* __restrict__ query,
    const float* __restrict__ db,
    int dim,
    int n_vecs,
    float* out
) {
    for (int v = 0; v < n_vecs; v++) {
        int base = v * dim;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int j = 0;

        for (; j + 16 <= dim; j += 16) {
            __m256 qa0 = _mm256_loadu_ps(query + j);
            __m256 db0 = _mm256_loadu_ps(db + base + j);
            __m256 qa1 = _mm256_loadu_ps(query + j + 8);
            __m256 db1 = _mm256_loadu_ps(db + base + j + 8);
            acc0 = _mm256_fmadd_ps(qa0, db0, acc0);
            acc1 = _mm256_fmadd_ps(qa1, db1, acc1);
        }

        for (; j + 8 <= dim; j += 8) {
            __m256 qa = _mm256_loadu_ps(query + j);
            __m256 dbv = _mm256_loadu_ps(db + base + j);
            acc0 = _mm256_fmadd_ps(qa, dbv, acc0);
        }

        __m256 sum = _mm256_add_ps(acc0, acc1);
        float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        float dot = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];

        for (; j < dim; j++)
            dot += query[j] * db[base + j];

        out[v] = dot;
    }
}
