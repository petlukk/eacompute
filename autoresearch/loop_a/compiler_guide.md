# Eä Compiler Guide for Loop A Agent

## Pipeline
Source (.ea) → Lexer → Parser → Desugar → Type Check → Codegen (LLVM 18) → .o/.so

## How to Add/Extend an Intrinsic

### Step 1: Type checking
File: `src/typeck/intrinsics.rs` (dispatch) or `src/typeck/intrinsics_simd.rs` (implementation)

To accept vector types, match on `Type::Vector`:
```rust
// Example from check_sqrt — accepts scalar and vector floats
match &arg_type {
    Type::F32 | Type::F64 | Type::FloatLiteral => Ok(arg_type),
    Type::Vector { elem, .. } if elem.is_float() => Ok(arg_type),
    _ => Err(...)
}
```

For two-argument intrinsics (like min/max), both args must have the same type:
```rust
match (&t1, &t2) {
    // existing scalar matches...
    (Type::Vector { elem: e1, width: w1 }, Type::Vector { elem: e2, width: w2 })
        if e1 == e2 && w1 == w2 && e1.is_float() => Ok(t1.clone()),
    _ => Err(...)
}
```

### Step 2: Codegen dispatch
File: `src/codegen/simd.rs`

The `is_simd_intrinsic()` function (line ~21) lists intrinsic names. The `compile_simd_call()` function (line ~64) routes names to compile functions. Both already include "min" and "max".

### Step 3: LLVM IR generation
File: `src/codegen/simd_math.rs`

For vector intrinsics, use the helper:
```rust
// self.llvm_vector_intrinsic_name("llvm.minnum", vec_ty)
// produces e.g. "llvm.minnum.v8f32" for f32x8
```

Full pattern for vector values:
```rust
BasicValueEnum::VectorValue(av) => {
    let bv = b.into_vector_value();
    let vec_ty = av.get_type();
    let base = if name == "min" { "llvm.minnum" } else { "llvm.maxnum" };
    let intrinsic_name = self.llvm_vector_intrinsic_name(base, vec_ty);
    let fn_type = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into()], false);
    let intrinsic = self.module.get_function(&intrinsic_name)
        .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
    let result = self.builder
        .build_call(intrinsic, &[av.into(), bv.into()], name)
        .map_err(|e| CompileError::codegen_error(e.to_string()))?
        .try_as_basic_value().left()
        .ok_or_else(|| CompileError::codegen_error(format!("{name} did not return a value")))?;
    Ok(result)
}
```

### Step 4: Tests
File: `tests/min_max_tests.rs` (or new test file)

End-to-end pattern:
```rust
#[test]
fn test_vector_min_f32x8() {
    assert_c_interop(
        r#"
        export func test(a: *f32, b: *f32, out: *mut f32) {
            let va: f32x8 = load(a, 0)
            let vb: f32x8 = load(b, 0)
            let vr: f32x8 = min(va, vb)
            store(out, 0, vr)
        }
        "#,
        r#"
        #include <stdio.h>
        #include <stdint.h>
        extern void test(const float*, const float*, float*);
        int main() {
            float a[] = {1,5,3,7,2,6,4,8};
            float b[] = {8,4,6,2,7,3,5,1};
            float out[8];
            test(a, b, out);
            for (int i = 0; i < 8; i++) printf("%.0f ", out[i]);
            printf("\n");
            return 0;
        }
        "#,
        "1 4 3 2 2 3 4 1",
    );
}
```

## File Map

| Change type | Files to modify |
|------------|----------------|
| Extend intrinsic to vectors | `typeck/intrinsics_simd.rs` + `codegen/simd_math.rs` + `tests/` |
| New intrinsic | `typeck/intrinsics.rs` (dispatch) + `typeck/intrinsics_simd.rs` + `codegen/simd.rs` (dispatch) + `codegen/simd_math.rs` + `tests/` |
| Codegen optimization | `codegen/simd_arithmetic.rs` or `codegen/simd_math.rs` |

## Hard Rules
- No file exceeds 500 lines (intrinsics_simd.rs is at 499 — split first if adding code)
- cargo fmt && cargo clippy clean
- Every feature proven by end-to-end test
- No stubs, TODOs, placeholders
