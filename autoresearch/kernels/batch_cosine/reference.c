// C AVX2 reference: batch cosine similarity
#include <immintrin.h>
#include <math.h>

void batch_cosine_ref(
    const float* __restrict__ query,
    const float* __restrict__ db,
    int dim,
    int n_vecs,
    float* out
) {
    // Precompute query norm
    __m256 qn0 = _mm256_setzero_ps();
    __m256 qn1 = _mm256_setzero_ps();
    int qi = 0;
    for (; qi + 16 <= dim; qi += 16) {
        __m256 qv0 = _mm256_loadu_ps(query + qi);
        __m256 qv1 = _mm256_loadu_ps(query + qi + 8);
        qn0 = _mm256_fmadd_ps(qv0, qv0, qn0);
        qn1 = _mm256_fmadd_ps(qv1, qv1, qn1);
    }
    for (; qi + 8 <= dim; qi += 8) {
        __m256 qv = _mm256_loadu_ps(query + qi);
        qn0 = _mm256_fmadd_ps(qv, qv, qn0);
    }
    __m256 qn_sum = _mm256_add_ps(qn0, qn1);
    float tmp[8];
    _mm256_storeu_ps(tmp, qn_sum);
    float query_sq = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; qi < dim; qi++)
        query_sq += query[qi] * query[qi];
    float query_norm = sqrtf(query_sq);

    if (query_norm == 0.0f) {
        for (int z = 0; z < n_vecs; z++) out[z] = 0.0f;
        return;
    }

    for (int v = 0; v < n_vecs; v++) {
        int base = v * dim;
        __m256 dot0 = _mm256_setzero_ps();
        __m256 dot1 = _mm256_setzero_ps();
        __m256 sq0 = _mm256_setzero_ps();
        __m256 sq1 = _mm256_setzero_ps();
        int j = 0;

        for (; j + 16 <= dim; j += 16) {
            __m256 qa0 = _mm256_loadu_ps(query + j);
            __m256 db0 = _mm256_loadu_ps(db + base + j);
            __m256 qa1 = _mm256_loadu_ps(query + j + 8);
            __m256 db1 = _mm256_loadu_ps(db + base + j + 8);
            dot0 = _mm256_fmadd_ps(qa0, db0, dot0);
            dot1 = _mm256_fmadd_ps(qa1, db1, dot1);
            sq0 = _mm256_fmadd_ps(db0, db0, sq0);
            sq1 = _mm256_fmadd_ps(db1, db1, sq1);
        }

        for (; j + 8 <= dim; j += 8) {
            __m256 qa = _mm256_loadu_ps(query + j);
            __m256 dbv = _mm256_loadu_ps(db + base + j);
            dot0 = _mm256_fmadd_ps(qa, dbv, dot0);
            sq0 = _mm256_fmadd_ps(dbv, dbv, sq0);
        }

        __m256 d_sum = _mm256_add_ps(dot0, dot1);
        __m256 s_sum = _mm256_add_ps(sq0, sq1);
        float dt[8], st[8];
        _mm256_storeu_ps(dt, d_sum);
        _mm256_storeu_ps(st, s_sum);
        float dot = dt[0]+dt[1]+dt[2]+dt[3]+dt[4]+dt[5]+dt[6]+dt[7];
        float sq_b = st[0]+st[1]+st[2]+st[3]+st[4]+st[5]+st[6]+st[7];

        for (; j < dim; j++) {
            dot += query[j] * db[base + j];
            sq_b += db[base + j] * db[base + j];
        }

        float db_norm = sqrtf(sq_b);
        if (db_norm == 0.0f)
            out[v] = 0.0f;
        else
            out[v] = dot / (query_norm * db_norm);
    }
}
