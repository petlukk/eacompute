# `ea bench`

Builds an `.ea` kernel and a C harness, runs the harness pinned to one core (Linux), captures JSONL measurements, wraps them with environment metadata, and (when a baseline exists) diffs against it.

## Usage

```
ea bench <manifest.toml> [--target=CPU] [--avx512|--fp16|--i8mm|--dotprod]
                         [--opt-level=N] [--update-baseline] [--no-diff]
                         [--out PATH]
```

A benchmark is a triple: an `.ea` kernel (`--lib`-compatible), a C harness that links against the kernel's shared library, and a manifest TOML that names them.

## Manifest schema

`*.bench.toml` files use this flat schema:

| key | type | required | meaning |
|-----|------|----------|---------|
| `name` | string | yes | Benchmark name (also the kernel's lib name) |
| `kernel` | string (path) | yes | Path to `.ea`, relative to the manifest |
| `harness` | string (path) | yes | Path to the C harness |
| `baseline` | string (path) | yes | Path to the committed baseline JSON |
| `arch` | array of string | yes | `"x86_64"` and/or `"aarch64"` — platforms where the kernel applies |
| `ea_flags` | array of string | no | Extra flags for the kernel build (e.g. `["--fp16"]`) |
| `cc_flags` | array of string | no | Extra flags for the harness build (default `["-O2"]`) |

Unknown keys are an error. Paths resolve relative to the manifest's parent directory; absolute paths pass through unchanged.

## Harness contract

A harness is a normal C program that links against the kernel's shared library. It must:

- **Write JSONL measurements to stdout, one per line.** Required keys: `kernel` (string), `median_ns` (integer). Optional: `p10_ns`, `p90_ns`, `n_inner`, `n_runs`. Any additional keys are passed through into the result JSON.
- **Send everything else to stderr.** Banners, verify-OK / verify-FAIL messages, debug output, sink values. `ea bench` relays harness stderr with a `[harness] ` prefix to its own stderr.
- **Exit 0 on success, non-zero on fatal error.** `ea bench` propagates a non-zero exit.

Example measurement line:

```json
{"kernel":"softmax_poly","median_ns":12345,"p10_ns":12200,"p90_ns":12600,"n_inner":200,"n_runs":10}
```

See `benchmarks/v1.11.0/exp_poly_f32_harness.c` for a complete template: deterministic LCG-filled input, warmup, median of N runs of M inner calls, volatile sink to defeat dead-code elimination, verify against a reference implementation, JSONL summary at the end.

## Output schema

`ea bench` emits a single JSON object (to stdout, or `--out PATH`):

```json
{
  "schema_version": 1,
  "name": "exp_poly_f32",
  "eacompute_version": "1.13.0",
  "git_sha": "f2ca320",
  "timestamp": "2026-05-14T10:23:00Z",
  "env": {
    "os": "linux", "arch": "x86_64", "host_cpu": "znver4",
    "target_cpu": "native", "target_features": "+avx512f,+avx512vl,+avx512bw",
    "opt_level": 3, "pinned": true
  },
  "measurements": [ /* harness JSONL lines, re-emitted */ ]
}
```

| field | source |
|-------|--------|
| `schema_version` | always `1` for this release |
| `name` | manifest `name` |
| `eacompute_version` | `CARGO_PKG_VERSION` of the running compiler |
| `git_sha` | `git rev-parse --short HEAD` at run time; `null` if not in a git tree |
| `timestamp` | ISO 8601 UTC |
| `env.host_cpu` | LLVM `TargetMachine::get_host_cpu_name()` |
| `env.target_cpu`, `target_features`, `opt_level` | resolved from CLI flags |
| `env.pinned` | `true` if `taskset` was used (Linux only) |

## Diff & baselines

If `manifest.baseline` exists, `ea bench` reads it and prints a per-kernel delta to stderr. The current regression threshold is **10%**, **warn-only** in v1.13.0 — regressions print `WARNING:` but the process still exits 0.

```
exp_poly_f32 (x86_64, native, opt=3):
  exp_only_libm   12345 ns  (baseline 12200 ns, +1.2%)
  exp_only_poly    4321 ns  (baseline  4350 ns, -0.7%)
  softmax_libm    98765 ns  (baseline 96000 ns, +2.9%)
  softmax_poly    32100 ns  (baseline 31500 ns, +1.9%)
WARNING: 0 regressions exceed 10% threshold (warn-only in v1.13.0).
```

If the baseline's `env.arch`, `env.target_features`, or `env.opt_level` don't match the current run, the diff is skipped and stderr says `baseline mismatch: ...`. Use `--update-baseline` to refresh.

If the baseline file doesn't exist (first run, or a freshly-added benchmark), stderr says `no baseline yet — run with --update-baseline to create one`.

## Adding a new benchmark

1. Write a kernel `foo_bench.ea` exporting `export func ...` measurement targets.
2. Write a C harness `foo_harness.c` following the harness contract above.
3. Write `foo.bench.toml` next to the kernel and harness.
4. Capture the baseline:
   ```
   ea bench foo.bench.toml --update-baseline
   ```
5. Commit kernel, harness, manifest, and baseline together.

## Platform notes

- **Pinning is Linux-only.** `ea bench` uses `taskset -c 0` when available. Mac / Windows runs report `pinned: false` in the result JSON, and measurements will be noisier.
- **Cross-platform manifests must use `arch = ["x86_64", "aarch64"]`** if the kernel + harness compile and run identically. Single-platform benchmarks use the matching single-element list.
- The `cc` binary can be overridden via the `CC` environment variable.
