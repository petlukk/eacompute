#include <stdint.h>
#include <string.h>

#define TABLE_SIZE 1024
#define KEY_STRIDE 64

void parse_aggregate_ref(
    const uint8_t *text,
    const int32_t *nl_pos,
    int32_t n,
    int32_t text_start,
    uint8_t *ht_keys,
    int32_t *ht_key_len,
    int32_t *ht_min,
    int32_t *ht_max,
    int32_t *ht_sum,
    int32_t *ht_count,
    int32_t *out_n_stations
) {
    int32_t n_stations = 0;

    for (int32_t i = 0; i < n; i++) {
        int32_t line_start = text_start;
        if (i > 0) {
            line_start = nl_pos[i - 1] + 1;
        }
        int32_t line_end = nl_pos[i];

        /* Find semicolon (backward scan) */
        int32_t semi_pos = line_end - 1;
        while (semi_pos >= line_start && text[semi_pos] != 59) {
            semi_pos--;
        }

        int32_t name_len = semi_pos - line_start;

        /* Hash station name: h = h * 31 + byte */
        int32_t h = 0;
        for (int32_t j = line_start; j < semi_pos; j++) {
            h = h * 31 + (int32_t)text[j];
        }

        /* Parse temperature */
        int32_t pos = semi_pos + 1;
        int32_t negative = 0;
        if (text[pos] == 45) { /* '-' */
            negative = 1;
            pos++;
        }

        int32_t int_val = 0;
        while (pos < line_end && text[pos] >= 48 && text[pos] <= 57) {
            int_val = int_val * 10 + ((int32_t)text[pos] - 48);
            pos++;
        }
        /* Skip '.' */
        pos++;
        int32_t dec_val = (int32_t)text[pos] - 48;
        int32_t temp = int_val * 10 + dec_val;
        if (negative) {
            temp = -temp;
        }

        /* Hash table probe (open addressing, linear probing) */
        int32_t slot = ((h % TABLE_SIZE) + TABLE_SIZE) % TABLE_SIZE;

        for (;;) {
            if (ht_key_len[slot] == 0) {
                /* Empty slot - insert */
                memcpy(ht_keys + slot * KEY_STRIDE, text + line_start, name_len);
                ht_key_len[slot] = name_len;
                ht_min[slot] = temp;
                ht_max[slot] = temp;
                ht_sum[slot] = temp;
                ht_count[slot] = 1;
                n_stations++;
                break;
            } else if (ht_key_len[slot] == name_len &&
                       memcmp(ht_keys + slot * KEY_STRIDE, text + line_start, name_len) == 0) {
                /* Match - update */
                if (temp < ht_min[slot]) ht_min[slot] = temp;
                if (temp > ht_max[slot]) ht_max[slot] = temp;
                ht_sum[slot] += temp;
                ht_count[slot]++;
                break;
            } else {
                /* Collision - linear probe */
                slot = (slot + 1) % TABLE_SIZE;
            }
        }
    }

    *out_n_stations = n_stations;
}
