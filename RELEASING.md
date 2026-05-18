# Releasing eacompute

Maintainer's release checklist. CI handles tag-triggered builds and GitHub
Release uploads via `.github/workflows/release.yml`; everything before the
tag push is manual on the `main` branch.

## Pre-tag

### 1. Refresh aarch64 baselines on the Pi 5

Pi 5 access (`peter@10.46.0.27`) is intermittent — do this **first** so a
Pi outage doesn't block the release. Assumes the Pi is already bootstrapped
(LLVM 18.1.8 at `~/llvm18`, eacompute checkout at `~/eacompute`).

```bash
ssh peter@10.46.0.27
cd ~/eacompute && git pull origin main && cargo build --release
./target/release/ea bench benchmarks/v1.11.0/fp16_kv.bench.toml
./target/release/ea bench benchmarks/v1.11.0/gather_compose_arm.bench.toml
```

Expect both to print `diff vs baseline: OK (within 10%)`. If a `WARNING:`
fires:

- Sub-percent moves around the existing baseline are run-to-run noise — don't update.
- Persistent >5% drift across multiple runs: re-run with `--update-baseline`,
  then `scp` the updated `*.baseline.json` to the dev host and commit.

`exp_poly_f32` is `arch = ["x86_64"]` and is skipped on aarch64 — see its
manifest. See `docs/src/reference/bench.md` for the schema and the
"how to add a new benchmark" recipe.

### 2. Refresh x86_64 baselines on dev host

```bash
./target/release/ea bench benchmarks/v1.11.0/exp_poly_f32.bench.toml
./target/release/ea bench benchmarks/v1.11.0/gather_compose.bench.toml
```

Same drift criteria as step 1. Update baselines only on persistent >5% drift.

### 3. Finalize CHANGELOG and version

- Rename the `## vX.Y.Z — UNRELEASED — <summary>` header to
  `## vX.Y.Z — YYYY-MM-DD — <summary>`.
- Cross-check the section against `git log --oneline vPREVIOUS..HEAD` —
  every commit should be reflected or intentionally omitted.
- Bump `version =` in `Cargo.toml`. `Cargo.lock` regenerates on next build.
- If breaking-change deprecations landed: ensure a
  `docs/migrations/vX.Y.Z.md` file exists per
  `docs/migrations/README.md`.
- If public API drifted intentionally: regenerate the snapshot via
  `cargo +nightly public-api --simplified > docs/public-api.txt`.
  CI's `public-api-check` job will fail otherwise.

### 4. Final test pass

```bash
cargo test --tests --features=llvm
```

Must be green locally before tagging. CI re-runs the same on three
platforms (x86 Linux, Linux ARM64, Windows) after the tag push.

## Tag

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Use `-a` (annotated tag) — release tags carry a tagger name, date, and
message, which is what `release.yml` and the GitHub Releases page expect.
A lightweight tag (`git tag vX.Y.Z` with no `-a`) is just a ref to a
commit and would lose that metadata.

`.github/workflows/release.yml` takes over: builds release binaries on
Linux x86_64, Linux aarch64, and Windows; uploads `.tar.gz` / `.zip`
artifacts to the GitHub Release page.

## Post-tag

- Confirm release artifacts appear on the GitHub Releases page (one per
  platform).
- Notify downstream consumers of any breaking-change items they need to
  act on: Olorin, eaclaw, Cougar, eachacha, eakv, ea-compiler-py.
