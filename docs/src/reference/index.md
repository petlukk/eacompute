# Specification

This page is the Eä language and library specification for `eacompute`. It
is the normative reference for what the compiler accepts and the semantics
of every built-in operation.

The specification is structured as the following parts:

- **[Type System](types.md)** — scalar, vector, pointer, and struct types;
  type rules including integer/float literal defaults and forbidden
  implicit conversions.
- **[All Intrinsics](intrinsics.md)** — every built-in function: memory
  (load/store/gather), math, reduction, vector, conversion, debug.
- **[CLI Reference](cli.md)** — the `ea` driver: `compile`, `bind`,
  `inspect`, `print-target`; target-feature flags (`--avx512`, `--fp16`,
  `--i8mm`, `--dotprod`); cross-compilation triple syntax.
- **[ARM / NEON](arm.md)** — AArch64-specific intrinsic surface,
  vector-width constraints, cross-compilation recipe, and the
  portable-kernels pattern.
- **[Binding Annotations](bindings.md)** — `out` qualifier, length-collapse
  via `[cap: n]`, and the `.ea.json` metadata schema consumed by language
  bindings.
- **[Python API](python-api.md)** — the `ea bind --python`-generated module
  surface.
- **[`ea bench`](bench.md)** — manifest schema, harness JSONL contract,
  baseline diff semantics.

## Normative vs informative

The seven documents above are **normative**: the compiler's behavior is
what they describe, and divergence is a bug in either the compiler or the
spec.

The [Guide](../guide/why-ea.md) and the
[Cookbook](../cookbook/numpy-comparison.md) are **informative** — teaching
material and worked examples. Where they conflict with the spec, the spec
wins.

## Stability and deprecation

The intrinsic and library API surface follows a deprecation cycle:

1. **Minor release.** The old name continues to work and emits a
   deprecation warning at every call site. The replacement is registered
   in `src/typeck/deprecations.rs`, and a migration entry lands in the
   upcoming-major-release file under `docs/migrations/`.
2. **Major release.** The old name is removed. Callers who ignored the
   warning now get an `unknown intrinsic` error; the migration file is the
   canonical recipe.

The `cargo public-api` CI gate prevents accidental Rust-side public-API
drift (snapshot at `docs/public-api.txt`).

## Version

This specification reflects eacompute on `main`. For the spec at a
specific tagged version, browse the same files from that tag.
