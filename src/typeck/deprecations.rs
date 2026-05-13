//! Intrinsic deprecation registry.
//!
//! A name in [`DEPRECATED_INTRINSICS`] still type-checks and code-generates
//! normally, but every call site is recorded as a [`DeprecationWarning`] on
//! the active `TypeChecker`. The compiler driver prints them to stderr; a
//! library consumer can pull them out via `TypeChecker::warnings()`.
//!
//! Policy (see `docs/migrations/README.md`): an intrinsic is deprecated for
//! at least one minor release before its name is removed in a major release.
//! The `advice` string must name the replacement spelling — recipients of
//! the warning should not have to read source to migrate.
//!
//! Tests inject synthetic entries via `TypeChecker::with_deprecations`.
//! The production table starts empty: the first real entries land with the
//! v1.12.0 monomorphic-rename batch (`sat_add` → `sat_add_i8x16`, etc.).

use crate::lexer::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct DeprecationInfo {
    /// Version where the deprecation took effect (e.g. `"1.12.0"`).
    pub since: &'static str,
    /// Replacement spelling and migration recipe.
    pub advice: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeprecationWarning {
    pub name: String,
    pub since: String,
    pub advice: String,
    pub span: Span,
}

impl std::fmt::Display for DeprecationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "warning: intrinsic '{}' is deprecated since {} — {} (at line {}, column {})",
            self.name, self.since, self.advice, self.span.start.line, self.span.start.column,
        )
    }
}

pub type DeprecationTable = &'static [(&'static str, DeprecationInfo)];

/// The production table of deprecated intrinsics.
///
/// Add entries here when renaming or removing an intrinsic: keep the old
/// name functional for at least one minor release cycle so callers see the
/// warning before the hard break in the next major release.
pub const DEPRECATED_INTRINSICS: DeprecationTable = &[
    (
        "sat_add",
        DeprecationInfo {
            since: "1.12.0",
            advice: "use the typed spelling: sat_add_i8x16 / sat_add_u8x16 / \
                     sat_add_i16x8 / sat_add_u16x8",
        },
    ),
    (
        "sat_sub",
        DeprecationInfo {
            since: "1.12.0",
            advice: "use the typed spelling: sat_sub_i8x16 / sat_sub_u8x16 / \
                     sat_sub_i16x8 / sat_sub_u16x8",
        },
    ),
    (
        "abs_diff",
        DeprecationInfo {
            since: "1.12.0",
            advice: "use the typed spelling: abs_diff_i8x16 / abs_diff_u8x16 / \
                     abs_diff_i16x8 / abs_diff_u16x8 / abs_diff_i32x4 / \
                     abs_diff_u32x4 (ARM-only)",
        },
    ),
];

pub fn lookup(table: DeprecationTable, name: &str) -> Option<&'static DeprecationInfo> {
    table.iter().find(|(n, _)| *n == name).map(|(_, info)| info)
}
