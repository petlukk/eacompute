# Vector Literals from Type Annotation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `let v: i16x8 = [a,b,c,d,e,f,g,h];` — the parser resolves the bracket expression to `Expr::Vector` using the declared type from the let binding, so `[...]` is never context-dependent at the type-checker level.

**Architecture:** The parser already has the declared type (`TypeAnnotation`) in `parse_let` before it calls `self.expression()`. We pass that type annotation into expression parsing. When the expression parser sees `[...]` with no suffix AND a vector type hint is active, it produces `Expr::Vector` (not `Expr::ArrayLiteral`). The type checker and codegen see `Expr::Vector` in both cases — no changes needed there.

**Tech Stack:** Rust, existing parser/AST/typeck/codegen infrastructure.

---

### Task 1: Add vector-literal test file with failing tests

**Files:**
- Create: `tests/vector_literal_tests.rs`

- [ ] **Step 1: Create test file with annotation-form tests**

```rust
#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_i32x4() {
        assert_output(
            r#"
export func main() {
    let v: i32x4 = [10, 20, 30, 40]
    println(extract(v, 0))
    println(extract(v, 1))
    println(extract(v, 2))
    println(extract(v, 3))
}
"#,
            "10\n20\n30\n40",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_f32x4() {
        assert_output(
            r#"
export func main() {
    let v: f32x4 = [1.5, 2.5, 3.5, 4.5]
    println(extract(v, 0))
    println(extract(v, 1))
}
"#,
            "1.5\n2.5",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_i16x8() {
        assert_output(
            r#"
export func main() {
    let v: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]
    println(extract(v, 0))
    println(extract(v, 7))
}
"#,
            "1\n8",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_annotation_arithmetic() {
        assert_output(
            r#"
export func main() {
    let a: i32x4 = [1, 2, 3, 4]
    let b: i32x4 = [10, 20, 30, 40]
    let c: i32x4 = a .+ b
    println(extract(c, 0))
    println(extract(c, 3))
}
"#,
            "11\n44",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_suffix_still_works() {
        assert_output(
            r#"
export func main() {
    let v: i32x4 = [1, 2, 3, 4]i32x4
    println(extract(v, 0))
}
"#,
            "1",
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test vector_literal_tests 2>&1 | head -30`
Expected: FAIL — parser produces `ArrayLiteral` which the type checker rejects with "array literals can only be used as shuffle indices"

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/vector_literal_tests.rs
git commit -m "test: add failing tests for vector literals from type annotation"
```

---

### Task 2: Thread type hint through parser

**Files:**
- Modify: `src/parser/mod.rs` (Parser struct — no changes needed, hint passed as parameter)
- Modify: `src/parser/statements.rs:95-122` (parse_let)
- Modify: `src/parser/expressions.rs:8,411-466` (expression, bracket parsing)

The parser needs to know the declared type when parsing a let-binding's RHS so that `[...]` without a suffix can produce `Expr::Vector` instead of `Expr::ArrayLiteral`.

- [ ] **Step 1: Add `expression_with_hint` method to parser**

In `src/parser/expressions.rs`, add a new entry point that accepts an optional type annotation hint and threads it to the bracket-parsing logic. The existing `expression()` delegates to `expression_with_hint(None)`.

```rust
// src/parser/expressions.rs — add after line 9 (after existing expression())

    pub(super) fn expression_with_hint(
        &mut self,
        _type_hint: Option<&crate::ast::TypeAnnotation>,
    ) -> crate::error::Result<Expr> {
        self.logical_or_with_hint(_type_hint)
    }
```

Update `expression()` to delegate:

```rust
    pub(super) fn expression(&mut self) -> crate::error::Result<Expr> {
        self.expression_with_hint(None)
    }
```

The hint only needs to reach the primary (atom) level where brackets are parsed. Thread it through the precedence chain. Each `logical_or`, `logical_and`, ... `unary`, `primary` method gets a `_with_hint` variant. However, since the hint is only consumed at the `primary` level (bracket parsing), the simplest approach is: only thread the hint to `primary`. The precedence chain calls always start at `primary` eventually — we just need the top-level call to pass the hint down.

**Simpler approach**: Since `[...]` is only parsed in `primary()`, and the precedence chain is `logical_or → logical_and → ... → unary → postfix → primary`, the hint must reach `primary`. But modifying every level is noisy. Instead, store the hint as a temporary field on Parser:

```rust
// src/parser/mod.rs — add field to Parser struct
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    let_type_hint: Option<TypeAnnotation>,  // hint for vector literal parsing
}
```

Initialize to `None` in `Parser::new`:
```rust
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0, let_type_hint: None }
    }
```

- [ ] **Step 2: Set the hint in `parse_let` before parsing the RHS expression**

In `src/parser/statements.rs`, after parsing the type annotation (line 111) and before calling `self.expression()` (line 113), set and clear the hint:

```rust
    fn parse_let(&mut self) -> crate::error::Result<Stmt> {
        let start = self.current_position();
        self.advance(); // consume 'let'
        let mutable = if self.check(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };
        let name_token =
            self.expect_kind(TokenKind::Identifier, "expected variable name after 'let'")?;
        let name = name_token.lexeme.clone();
        self.expect_kind(
            TokenKind::Colon,
            "expected ':' after variable name (type annotation required)",
        )?;
        let ty = self.parse_type()?;
        self.expect_kind(TokenKind::Equals, "expected '=' after type annotation")?;
        self.let_type_hint = Some(ty.clone());
        let value = self.expression()?;
        self.let_type_hint = None;
        let end = value.span().end.clone();
        Ok(Stmt::Let {
            name,
            ty,
            value,
            mutable,
            span: Span::new(start, end),
        })
    }
```

- [ ] **Step 3: Use the hint in bracket parsing to produce `Expr::Vector`**

In `src/parser/expressions.rs`, in the bracket-parsing block (around line 464), before falling back to `ArrayLiteral`, check the hint:

Replace this code (lines 464-466):
```rust
            // No type suffix — it's an array literal (used for shuffle masks etc.)
            let end = self.previous_position();
            return Ok(Expr::ArrayLiteral(elements, Span::new(start, end)));
```

With:
```rust
            // No type suffix — check if let-binding declares a vector type
            let end = self.previous_position();
            if let Some(ref hint) = self.let_type_hint {
                if let TypeAnnotation::Vector { elem, width, span: ty_span } = hint {
                    return Ok(Expr::Vector {
                        elements,
                        ty: TypeAnnotation::Vector {
                            elem: elem.clone(),
                            width: *width,
                            span: ty_span.clone(),
                        },
                        span: Span::new(start, end),
                    });
                }
            }
            return Ok(Expr::ArrayLiteral(elements, Span::new(start, end)));
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --test vector_literal_tests 2>&1 | tail -20`
Expected: All 5 tests PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `cargo test --tests --features=llvm 2>&1 | tail -5`
Expected: All ~420+ tests pass. Shuffle masks (which use `ArrayLiteral` without a vector let-binding hint) continue to work because `let_type_hint` is `None` in those contexts.

- [ ] **Step 6: Run clippy and fmt**

Run: `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -10`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add src/parser/mod.rs src/parser/statements.rs src/parser/expressions.rs
git commit -m "feat: vector literals from let-binding type annotation

let v: i16x8 = [1,2,3,4,5,6,7,8] now works — parser resolves
bracket expr to Expr::Vector using the declared type. No changes
to type checker or codegen. Suffix form still works."
```

---

### Task 3: Add error-case tests

**Files:**
- Modify: `tests/vector_literal_tests.rs`

- [ ] **Step 1: Add element count mismatch test**

```rust
    #[test]
    fn test_vector_literal_wrong_count() {
        let source = r#"
export func main() {
    let v: i32x4 = [1, 2, 3]
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts);
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        assert!(
            format!("{err:?}").contains("4 elements") || format!("{err:?}").contains("expects"),
            "should report element count mismatch, got: {err:?}"
        );
    }
```

- [ ] **Step 2: Add wrong element type test**

```rust
    #[test]
    fn test_vector_literal_wrong_elem_type() {
        let source = r#"
export func main() {
    let v: i32x4 = [1.0, 2.0, 3.0, 4.0]
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts);
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        assert!(
            format!("{err:?}").contains("expected i32") || format!("{err:?}").contains("element"),
            "should report type mismatch, got: {err:?}"
        );
    }
```

- [ ] **Step 3: Add test that bare array literal still errors in non-let context**

```rust
    #[test]
    fn test_bare_array_literal_still_errors() {
        let source = r#"
export func f(v: i32x4) -> i32x4 {
    return [1, 2, 3, 4]
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts);
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        assert!(
            format!("{err:?}").contains("shuffle"),
            "bare array literal in return should still error, got: {err:?}"
        );
    }
```

- [ ] **Step 4: Run the error tests**

Run: `cargo test --test vector_literal_tests 2>&1 | tail -15`
Expected: All tests pass (including both happy-path and error tests)

- [ ] **Step 5: Commit**

```bash
git add tests/vector_literal_tests.rs
git commit -m "test: error cases for vector literal annotation form"
```

---

### Task 4: Add u8x16 and i8x16 annotation-form tests

These types are important for the NEON/quantization kernels in Olorin.

**Files:**
- Modify: `tests/vector_literal_tests.rs`

- [ ] **Step 1: Add i8x16 and u8x16 tests**

```rust
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_u8x16() {
        assert_output(
            r#"
export func main() {
    let v: u8x16 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    println(extract(v, 0))
    println(extract(v, 15))
}
"#,
            "10\n160",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_i8x16() {
        assert_output(
            r#"
export func main() {
    let v: i8x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    println(extract(v, 0))
    println(extract(v, 15))
}
"#,
            "1\n16",
        );
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test --test vector_literal_tests 2>&1 | tail -15`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/vector_literal_tests.rs
git commit -m "test: vector literal annotation form for i8x16 and u8x16"
```
