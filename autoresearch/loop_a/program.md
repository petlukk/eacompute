# Eä Compiler Optimization

You are modifying the Eä compiler (Rust) to improve codegen quality. Your changes are validated against 421 existing tests and 5 benchmark kernels.

## Your Task

Implement the requested compiler change. You MUST output a HYPOTHESIS line and then one or more FILE + diff blocks.

## Rules

1. Only valid Rust. Must compile, pass clippy, pass all 421 tests.
2. One minimal change per iteration. State your hypothesis clearly.
3. Diffs must apply cleanly with `git apply`.
4. Include test additions when adding new functionality.
5. Do not modify test infrastructure (`tests/common/mod.rs`).
6. No file may exceed 500 lines after your change.
7. No TODOs, stubs, or placeholders.

## Output Format

Your output MUST contain:

1. A line starting with HYPOTHESIS: followed by what you are changing and why
2. One or more FILE: + diff blocks:

HYPOTHESIS: Extend min/max to accept f32x4 vectors in type checker

FILE: src/typeck/intrinsics_simd.rs
```diff
@@ -445,6 +445,8 @@
 context line
-old line
+new line
 context line
```

FILE: tests/min_max_tests.rs
```diff
@@ -183,0 +184,15 @@
+    new test code
```

Do NOT omit the HYPOTHESIS line. Do NOT output partial diffs.
