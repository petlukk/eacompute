//! Deprecation-warning infrastructure tests.
//!
//! The production `DEPRECATED_INTRINSICS` table is empty as of v1.12.0-dev;
//! these tests use `TypeChecker::with_deprecations` to inject a synthetic
//! entry against `abs_diff` (a real, working intrinsic) so we exercise the
//! full path: source -> tokens -> parse -> typeck -> warning recorded.
//!
//! Once a real intrinsic ships with deprecation (the v1.12.0 monomorphic-
//! rename batch), an end-to-end test against `compile_with_options` should
//! be added that asserts the warning appears on stderr.

use ea_compiler::typeck::{DeprecationInfo, DeprecationTable, TypeChecker};

static SYNTHETIC_TABLE: DeprecationTable = &[(
    "abs_diff",
    DeprecationInfo {
        since: "1.12.0",
        advice: "use abs_diff_i8x16 / abs_diff_u8x16 / abs_diff_i16x8 / \
                 abs_diff_u16x8 / abs_diff_i32x4 / abs_diff_u32x4 instead",
    },
)];

const SOURCE_USING_DEPRECATED: &str =
    "export func f(a: u8x16, b: u8x16) -> u8x16 { return abs_diff(a, b) }\n";

const SOURCE_USING_NONE: &str = "export func f(a: u8x16, b: u8x16) -> u8x16 { return a .+ b }\n";

fn pipeline(source: &str) -> Vec<ea_compiler::DeprecationWarning> {
    let tokens = ea_compiler::tokenize(source).expect("tokenize");
    let stmts = ea_compiler::parse(tokens).expect("parse");
    let stmts = ea_compiler::desugar(stmts).expect("desugar");
    let mut tc = TypeChecker::with_deprecations(SYNTHETIC_TABLE);
    tc.check_program(&stmts).expect("typeck");
    tc.warnings()
}

#[test]
fn deprecated_intrinsic_emits_warning() {
    let warnings = pipeline(SOURCE_USING_DEPRECATED);
    assert_eq!(warnings.len(), 1, "expected exactly one warning");
    let w = &warnings[0];
    assert_eq!(w.name, "abs_diff");
    assert_eq!(w.since, "1.12.0");
    assert!(
        w.advice.contains("abs_diff_i8x16"),
        "advice should name a replacement: {}",
        w.advice
    );
    assert!(w.span.start.line >= 1);
}

#[test]
fn warning_display_includes_name_and_span() {
    let warnings = pipeline(SOURCE_USING_DEPRECATED);
    let rendered = format!("{}", warnings[0]);
    assert!(rendered.contains("abs_diff"), "rendered: {rendered}");
    assert!(rendered.contains("1.12.0"), "rendered: {rendered}");
    assert!(rendered.contains("line"), "rendered: {rendered}");
}

#[test]
fn non_deprecated_call_emits_no_warning() {
    let warnings = pipeline(SOURCE_USING_NONE);
    assert!(warnings.is_empty(), "got unexpected warnings: {warnings:?}");
}

#[test]
fn production_table_starts_empty() {
    // Sanity check on policy: until the v1.12.0 rename batch lands, the
    // production deprecation table should have no entries.
    assert!(
        ea_compiler::typeck::DEPRECATED_INTRINSICS.is_empty(),
        "DEPRECATED_INTRINSICS should be empty at v1.12.0-dev; \
         when adding entries, update this test and docs/migrations/"
    );
}

#[test]
fn default_typechecker_emits_no_warnings_for_known_intrinsic() {
    // With the default (production) table, a working intrinsic must not
    // record a warning — guards against the recorder firing on every call.
    let tokens = ea_compiler::tokenize(SOURCE_USING_DEPRECATED).unwrap();
    let stmts = ea_compiler::parse(tokens).unwrap();
    let stmts = ea_compiler::desugar(stmts).unwrap();
    let mut tc = TypeChecker::new();
    tc.check_program(&stmts).unwrap();
    assert!(tc.warnings().is_empty());
}

#[test]
fn shadowed_user_function_does_not_warn() {
    // If a user defines a function named like a deprecated intrinsic AND
    // the typechecker resolves to the user function (e.g. via the `abs`
    // shadowing rule), no deprecation warning should fire. abs_diff
    // doesn't currently support user-shadowing, so this test asserts the
    // invariant on the recording site: only intrinsic dispatches warn.
    let src = "func helper(a: u8x16, b: u8x16) -> u8x16 { return a .+ b }\n\
               export func f(a: u8x16, b: u8x16) -> u8x16 { return helper(a, b) }\n";
    let warnings = pipeline(src);
    assert!(warnings.is_empty(), "got: {warnings:?}");
}
