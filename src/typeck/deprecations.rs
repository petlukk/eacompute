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
/// Empty as of v1.12.0-dev. Add entries here when removing an intrinsic name
/// in a future release: keep it functional for one minor cycle so callers see
/// the warning before the hard break.
pub const DEPRECATED_INTRINSICS: DeprecationTable = &[];

pub fn lookup(table: DeprecationTable, name: &str) -> Option<&'static DeprecationInfo> {
    table.iter().find(|(n, _)| *n == name).map(|(_, info)| info)
}
