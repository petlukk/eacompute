// Regression: `let` bindings inside a loop body must alloca in the function
// entry block, not the loop body. An alloca in a loop body is a dynamic stack
// allocation that re-executes every iteration and is never reclaimed until the
// function returns, so the stack grows per iteration. mem2reg/SROA only seed
// promotion from the entry block, so loop-body allocas also survive past -O0.
//
// This is the root cause of the eacompute stack-overflow found via Olorin's
// log_level_scan.ea: a SIMD loop body with ~50 u8x16 `let` bindings overflowed
// the 8 MB main-thread stack on >=~1 MB inputs (~8 bytes of stack per input
// byte). The frontend emits unoptimized IR here, so the assertion is exact and
// opt-level independent.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    /// Assert that every `alloca` in `ir` lives in a basic block named `entry`.
    /// Tracks the current block label while scanning the IR text.
    fn assert_all_allocas_in_entry(ir: &str) {
        let mut current_block = String::new();
        for line in ir.lines() {
            let trimmed = line.trim();
            // Basic-block label lines look like `while_body71:` or
            // `while_body71:    ; preds = ...`. Strip any `; ...` comment first
            // (the `preds = ...` comment contains `=`), then match `<label>:`
            // with nothing but whitespace before the colon.
            let code = trimmed.split(';').next().unwrap_or("").trim();
            if let Some(label) = code.strip_suffix(':') {
                let is_label = !label.is_empty()
                    && label
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
                if is_label {
                    current_block = label.to_string();
                    continue;
                }
            }
            if trimmed.contains("= alloca") {
                assert_eq!(
                    current_block, "entry",
                    "alloca emitted in block `{current_block}`, expected `entry`:\n  {trimmed}\n\nFull IR:\n{ir}"
                );
            }
        }
    }

    #[test]
    fn let_bindings_in_while_body_alloca_in_entry() {
        // Mirrors the log_level_scan.ea pattern: many vector `let`s per chunk.
        let source = r#"
            export func k(text: *u8, n: i32, out: *mut i32) {
                let mut i: i32 = 0
                while i + 16 <= n {
                    let a: u8x16 = load(text, i)
                    let b: u8x16 = load(text, i + 1)
                    let c: u8x16 = a .& b
                    let d: u8x16 = a .| b
                    let e: u8x16 = select(a .== b, c, d)
                    let f: u8x16 = c .& d
                    out[0] = out[0] + to_i32(reduce_add(e))
                    out[1] = out[1] + to_i32(reduce_add(f))
                    i = i + 16
                }
            }
        "#;
        let ir = compile_to_ir(source);
        assert_all_allocas_in_entry(&ir);
    }

    #[test]
    fn nested_while_let_bindings_alloca_in_entry() {
        // A `let` inside a nested loop is the worst case: per-iteration growth
        // multiplied across both loop trip counts.
        let source = r#"
            export func k(text: *u8, n: i32, out: *mut i32) {
                let mut i: i32 = 0
                while i + 16 <= n {
                    let mut j: i32 = 0
                    while j < 4 {
                        let v: u8x16 = load(text, i)
                        out[0] = out[0] + to_i32(reduce_add(v))
                        j = j + 1
                    }
                    i = i + 16
                }
            }
        "#;
        let ir = compile_to_ir(source);
        assert_all_allocas_in_entry(&ir);
    }
}
