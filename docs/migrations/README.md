# Migration Guides

One file per breaking release. Each file documents:

1. **What changed** — the breaking-change list with minimal context.
2. **Migration recipe** — exact source-level rewrite for each item.
3. **Deprecation cycle** — when the warning first appeared and when the old
   spelling will be removed.

## Policy

A breaking change to the intrinsic surface or library API must go through:

1. **Deprecation release (minor)**: the old name continues to work but emits
   a `warning: intrinsic '...' is deprecated since X.Y.Z` line at every call
   site. The replacement spelling is registered in
   `src/typeck/deprecations.rs::DEPRECATED_INTRINSICS`, and a migration entry
   is added to the upcoming major-release file in this directory.
2. **Removal release (major)**: the old name is deleted. Callers who ignored
   the deprecation warning now get `unknown intrinsic '...'` at compile time;
   the migration file in this directory is the canonical recipe.

The one-minor-release minimum exists so consumers (`Olorin`, `eaclaw`,
`Cougar`, `eachacha`, `eakv`, `ea-compiler-py`) get at least one CI cycle of
loud warnings before the hard break.

## Index

- [`v1.11.0.md`](v1.11.0.md) — retroactive: `maddubs_i32` removed in favor
  of `maddubs_i16` + `madd_i16` chain. (Shipped without a deprecation cycle;
  documented here for historical migration support.)
- [`v1.12.0.md`](v1.12.0.md) — upcoming: monomorphic-rename batch for
  `sat_add` / `sat_sub` / `abs_diff`. Deprecation cycle begins v1.12.0.

## Authoring a new migration file

Use this template:

```markdown
# Migration to vX.Y.Z

## Summary

One sentence per breaking item.

## Breaking changes

### Item: `old_name` removed (use `new_name`)

- **Why:** brief justification.
- **Deprecation cycle:** warning emitted since `vA.B.C`; removed in `vX.Y.Z`.
- **Migration recipe:**

  ```ea
  // Before:
  let x = old_name(args)
  // After:
  let x = new_name(args)
  ```

- **Affects:** known downstream consumers if any.
```
