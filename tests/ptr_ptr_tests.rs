#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    fn test_ptr_ptr_read() {
        assert_c_interop(
            r#"
            export func sum_rows(rows: **f32, n_rows: i32, cols: i32, out: *mut f32) {
                let mut i: i32 = 0
                while i < n_rows {
                    let row: *f32 = rows[i]
                    let mut j: i32 = 0
                    while j < cols {
                        out[j] = out[j] + row[j]
                        j = j + 1
                    }
                    i = i + 1
                }
            }
            "#,
            r#"
            #include <stdio.h>
            extern void sum_rows(const float**, int, int, float*);
            int main() {
                float r0[] = {1.0, 2.0, 3.0};
                float r1[] = {4.0, 5.0, 6.0};
                const float* rows[] = {r0, r1};
                float out[3] = {0, 0, 0};
                sum_rows(rows, 2, 3, out);
                printf("%.0f %.0f %.0f\n", out[0], out[1], out[2]);
                return 0;
            }
            "#,
            "5 7 9",
        );
    }

    #[test]
    fn test_ptr_ptr_mut() {
        assert_c_interop(
            r#"
            export func fill_rows(rows: **mut f32, n_rows: i32, cols: i32) {
                let mut i: i32 = 0
                while i < n_rows {
                    let row: *mut f32 = rows[i]
                    let mut j: i32 = 0
                    while j < cols {
                        row[j] = to_f32(i * cols + j)
                        j = j + 1
                    }
                    i = i + 1
                }
            }
            "#,
            r#"
            #include <stdio.h>
            extern void fill_rows(float**, int, int);
            int main() {
                float r0[3], r1[3];
                float* rows[] = {r0, r1};
                fill_rows(rows, 2, 3);
                printf("%.0f %.0f %.0f %.0f %.0f %.0f\n",
                    r0[0], r0[1], r0[2], r1[0], r1[1], r1[2]);
                return 0;
            }
            "#,
            "0 1 2 3 4 5",
        );
    }

    #[test]
    fn test_ptr_ptr_i32() {
        assert_c_interop(
            r#"
            export func sum_columns(data: **i32, n_rows: i32, n_cols: i32, out: *mut i32) {
                let mut j: i32 = 0
                while j < n_cols {
                    let mut total: i32 = 0
                    let mut i: i32 = 0
                    while i < n_rows {
                        let row: *i32 = data[i]
                        total = total + row[j]
                        i = i + 1
                    }
                    out[j] = total
                    j = j + 1
                }
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void sum_columns(const int32_t**, int32_t, int32_t, int32_t*);
            int main() {
                int32_t r0[] = {1, 2, 3};
                int32_t r1[] = {10, 20, 30};
                int32_t r2[] = {100, 200, 300};
                const int32_t* rows[] = {r0, r1, r2};
                int32_t out[3];
                sum_columns(rows, 3, 3, out);
                printf("%d %d %d\n", out[0], out[1], out[2]);
                return 0;
            }
            "#,
            "111 222 333",
        );
    }
}
