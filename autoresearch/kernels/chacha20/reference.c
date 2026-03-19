/*
 * chacha20_ref.c — Generic ChaCha20 reference (RFC 7539), no SIMD intrinsics.
 * Compile: cc -O3 -shared -fPIC -o libchacha20_ref.so chacha20_ref.c
 */
#include <stdint.h>
#include <string.h>

#define ROTL32(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define QR(a, b, c, d)          \
    a += b; d ^= a; d = ROTL32(d, 16); \
    c += d; b ^= c; b = ROTL32(b, 12); \
    a += b; d ^= a; d = ROTL32(d,  8); \
    c += d; b ^= c; b = ROTL32(b,  7);

static void chacha20_block_ref(const uint32_t key[8], const uint32_t nonce[3],
                               uint32_t counter, uint32_t out[16])
{
    /* Initial state */
    uint32_t s[16];
    s[ 0] = 0x61707865; s[ 1] = 0x3320646e;
    s[ 2] = 0x79622d32; s[ 3] = 0x6b206574;
    s[ 4] = key[0]; s[ 5] = key[1]; s[ 6] = key[2]; s[ 7] = key[3];
    s[ 8] = key[4]; s[ 9] = key[5]; s[10] = key[6]; s[11] = key[7];
    s[12] = counter; s[13] = nonce[0]; s[14] = nonce[1]; s[15] = nonce[2];

    /* Working copy */
    uint32_t w[16];
    memcpy(w, s, sizeof(w));

    /* 20 rounds (10 double-rounds) */
    for (int i = 0; i < 10; i++) {
        /* Column round */
        QR(w[ 0], w[ 4], w[ 8], w[12])
        QR(w[ 1], w[ 5], w[ 9], w[13])
        QR(w[ 2], w[ 6], w[10], w[14])
        QR(w[ 3], w[ 7], w[11], w[15])
        /* Diagonal round */
        QR(w[ 0], w[ 5], w[10], w[15])
        QR(w[ 1], w[ 6], w[11], w[12])
        QR(w[ 2], w[ 7], w[ 8], w[13])
        QR(w[ 3], w[ 4], w[ 9], w[14])
    }

    /* Add initial state */
    for (int i = 0; i < 16; i++)
        out[i] = w[i] + s[i];
}

void chacha20_encrypt_ref(const uint32_t key[8], const uint32_t nonce[3],
                          uint32_t counter, const uint8_t *pt, uint8_t *ct,
                          int len)
{
    uint32_t ks[16];
    int offset = 0;

    while (offset < len) {
        chacha20_block_ref(key, nonce, counter, ks);

        int block_len = len - offset;
        if (block_len > 64)
            block_len = 64;

        const uint8_t *ks_bytes = (const uint8_t *)ks;
        for (int i = 0; i < block_len; i++)
            ct[offset + i] = pt[offset + i] ^ ks_bytes[i];

        offset += 64;
        counter++;
    }
}
