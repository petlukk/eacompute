# v1.11.0 Test Coverage Audit

Phase 4 of the pre-merge audit. Per-intrinsic coverage matrix across the
63 net-new intrinsics catalogued in `inventory.md`, with a gap-filling
pass for the only family that had zero tests (`ptr_as_*`).

Branch: `feat/i8mm-intrinsics` (3559c51 + this phase). Baseline: 765
tests. Compiled: 2026-05-13.

## Coverage matrix

Columns:

- **E2E**: an end-to-end test compiles + runs the intrinsic and checks
  observable output (`assert_output`, `assert_c_interop`, or
  shared-lib equivalents).
- **IR**: an IR-level regression guard pins the LLVM intrinsic / IR
  pattern that the codegen must emit (e.g. `aarch64.neon.smmla`).
- **Negative**: a typeck rejection test with `check_types(...).is_err()`
  or `try_compile(...).unwrap_err()` plus a `.contains(...)` assertion
  on the message.
- **Cross-arch**: a wrong-target test asserting the intrinsic is
  rejected on the architecture it isn't built for. Marked N/A for
  cross-platform intrinsics that work on every supported target.

Legend: `Y` test exists, `-` missing (gap), `N/A` not applicable.

### Cross-platform (29 intrinsics + 9 widen variants)

Most cross-platform rows can be covered by the family file alone. The
"Cross-arch" column is `N/A` for genuinely-portable intrinsics; it's
`Y` for cross-platform names whose wider variants are arch-gated by
register width (those wider forms appear in the x86-only table).

| Family | Members | E2E | IR | Negative | Cross-arch |
|---|---|---|---|---|---|
| `abs` (scalar + vec) | 1 | Y (`abs_tests.rs`) | Y (`fabs` IR check) | N/A | N/A |
| `bitcast_*` (4 variants) | `bitcast_i8x{16,32}`, `bitcast_i32x{4,8}` | Y (`phase_b_ext.rs`) | N/A | N/A | N/A |
| AVX-512 lane family — `concat_*` (5) | `concat_i8x{16,32}`, `concat_u8x{16,32}`, `concat_i32x8`, `concat_f32x8` | Y (`phase_b_avx512_lane.rs`) | N/A | N/A | Y (`phase_b_avx512_arm_safety.rs` for ARM-side 256→512 width rejection) |
| AVX-512 lane family — `lo*_*` / `hi*_*` (10) | `lo/hi128_{i,u}8x32`, `lo/hi256_{i,u}8x64`, `lo/hi256_i32x16`, `lo/hi256_f32x16` | Y (`phase_b_avx512_lane.rs`) | N/A | N/A | Y (`phase_b_avx512_arm_safety.rs`) |
| AVX-512 lane family — `shuffle_i32x{8,16}` | 2 | Y (`phase_b_avx512_lane.rs`) | N/A | N/A | Y (`phase_b_avx512_arm_safety.rs`) |
| AVX-512 lane family — `blend_i32` | 1 | Y (`phase_b_avx512_lane.rs`) | N/A | N/A | Y (`phase_b_avx512_arm_safety.rs`) |
| AVX-512 lane family — `bcast_*_pairs_*` (4) | `bcast_{even,odd}_pairs_i32x{8,16}` | Y (`phase_b_avx512_lane.rs`) | N/A | N/A | Y (`phase_b_avx512_arm_safety.rs`) |
| `f32x{4,8}_from_scalars` | 2 | Y (`phase14_arm_neon.rs`) | N/A | N/A | N/A |
| byte-shift family | `bslli/bsrli_i8x{16,32}` | Y (`phase14_byteshift.rs`) | N/A | N/A | Y (`phase14_byteshift.rs`, x32 ARM-reject) |
| `cvt_f16_f32` (3 widths) / `cvt_f32_f16` (2) | — | Y (`phase14_arm_ext.rs`, `phase_b_avx2.rs`, `phase_b_avx512_dotprod.rs`) | N/A | N/A | Y (`phase_b_avx2.rs` x86-only width gates) |
| `round_f32x{4,8}_i32x{4,8}` | 2 | Y (`phase14_pack.rs`) | N/A | N/A | Y (`phase14_pack.rs` ARM wide-vec reject) |
| `pack_sat_i16x8` / `pack_sat_i32x4` | 2 | Y (`phase14_pack.rs`) | N/A | N/A | N/A |
| `pack_usat_i16x8` / `pack_usat_i32x4` | 2 | Y (`phase14_pack_unsigned.rs`) | N/A | N/A | N/A |
| `sat_add` / `sat_sub` | 2 | Y (`phase14_sat.rs`) | N/A | N/A | N/A |
| `exp_poly_f32` | 1 | Y (`phase14_exp_poly.rs`) | Y (no `@llvm.exp`, fma count, ldexp pattern) | Y (rejects scalar, rejects f64) | N/A |
| `to_f16` (scalar) | 1 | Y (`phase14_arm_fp16.rs`, `rmsnorm_f16.ea`) | N/A | N/A | N/A |
| `to_i16` (scalar) | 1 | Y (`phase_b_dotprod.rs`) | N/A | N/A | N/A |
| `widen_u8_u16` | 1 | Y (`phase_b_avx2.rs`) | N/A | N/A | N/A |
| `widen_*_*_offset` family | 9 | Y (`phase14_widen.rs`) | N/A | N/A | N/A |
| `ptr_as_*` family | 10 (i8/u8/i16/u16/i32/u32/i64/u64/f32/f64) | **Y (`ptr_as_tests.rs` — NEW THIS PHASE)** | N/A | **Y (`ptr_as_tests.rs` — NEW)** | N/A |

### x86-only (6 intrinsics)

| Intrinsic | E2E | IR | Negative | Cross-arch (ARM rejection) |
|---|---|---|---|---|
| `madd_i16` (3 widths) | Y (`phase_b_avx2.rs`, `phase_b_avx512_dotprod.rs`, `phase_b_dotprod.rs`) | N/A | N/A | Y (`phase14_arm.rs:107`) |
| `hadd_i16` (2 widths) | Y (`phase_b_dotprod.rs`) | N/A | N/A | Y (`phase14_arm.rs:122`) |
| `pack_sat_i16x16` | Y (`phase14_pack.rs`) | N/A | N/A | Y (`phase14_pack.rs`) |
| `pack_sat_i32x8` | Y (`phase14_pack.rs`) | N/A | N/A | Y (`phase14_pack.rs`) |
| `pack_usat_i16x16` | Y (`phase14_pack_unsigned.rs`) | N/A | N/A | Y (`phase14_pack_unsigned.rs`) |
| `pack_usat_i32x8` | Y (`phase14_pack_unsigned.rs`) | N/A | N/A | Y (`phase14_pack_unsigned.rs`) |

### ARM-only (11 intrinsics + f16 vector arithmetic family)

| Intrinsic | E2E | IR | Negative | Cross-arch (x86 rejection) |
|---|---|---|---|---|
| `abs_diff` (6 type combos) | Y (`phase14_arm_neon.rs`) | N/A | N/A | Y (`phase14_arm_neon.rs::test_x86_rejects_abs_diff`) |
| `addp_i16` / `addp_i32` | Y (`phase14_arm_neon.rs`) | N/A | N/A | Y (`phase14_arm_neon.rs::test_arm_addp_i32_rejected_on_x86`) |
| `wmul_i16` / `wmul_u16` / `wmul_i32` / `wmul_u32` | Y (`phase14_arm_neon.rs`) | N/A | N/A | Y (`phase14_arm_neon.rs::test_x86_rejects_wmul_i16`) |
| `vdot_lane_i32` | Y (`phase14_arm.rs`, `phase_b_ext.rs`) | Y (`aarch64.neon.sdot` + `shufflevector`) | Y (out-of-range lane, wrong type) | Y (`phase14_arm.rs::test_arm_rejects_vdot_lane_i32_without_dotprod`) |
| `smmla_i32` | Y (`phase14_arm.rs`) | Y (`aarch64.neon.smmla`) | Y (wrong arg types) | Y (`phase14_arm.rs::test_x86_rejects_smmla_i32`, `test_arm_rejects_smmla_i32_without_i8mm`) |
| `ummla_i32` | Y (`phase14_arm_i8mm.rs`) | Y (`aarch64.neon.ummla`) | Y | Y (`phase14_arm_i8mm.rs`) |
| `usmmla_i32` | Y (`phase14_arm_i8mm.rs`) | Y (`aarch64.neon.usmmla`) | Y | Y (`phase14_arm_i8mm.rs`) |
| f16 vector arithmetic (`+`, `-`, `*`, `/` on `f16xN`) | Y (`phase14_arm_fp16.rs`) | N/A | N/A | Y (`phase_b_avx512_arm_safety.rs::test_fp16_on_x86_is_rejected`) |
| f16 `splat`/`load`/`store`/`fma`/`reduce_*` | Y (`phase14_arm_fp16.rs`) | N/A | N/A | Y (same as above, +fullfp16 gate is library-mirrored) |

## Gaps identified

A `-` mark in the matrix flags a gap. After this phase, only one category
of gap remained at the start:

- **`ptr_as_*` (10 intrinsics, all 4 columns missing E2E + Negative).**
  Flagged by Phase 1 inventory; no test file existed.

All other branch-touched intrinsics had at least E2E coverage. IR-level
guards are deliberately limited to perf-critical intrinsics
(`exp_poly_f32`, `vdot*`, `smmla*`) per the audit plan; the lane-family
intrinsics are well-covered by E2E behavior tests and don't warrant an
IR pin, since they don't have a "could silently fall back to a slow path"
failure mode the way `exp_poly_f32` did.

## Gaps filled in this phase

`tests/ptr_as_tests.rs` (new file, 13 tests, +303 lines):

- E2E round-trip for each typed variant:
  - `test_ptr_as_i8_roundtrip` (i8/u8 byte reinterpretation, signed read)
  - `test_ptr_as_u8_roundtrip` (write through `*mut u8` view)
  - `test_ptr_as_i16_roundtrip`
  - `test_ptr_as_u16_roundtrip`
  - `test_ptr_as_i32_roundtrip`
  - `test_ptr_as_u32_roundtrip`
  - `test_ptr_as_i64_roundtrip`
  - `test_ptr_as_u64_roundtrip`
  - `test_ptr_as_f32_roundtrip`
  - `test_ptr_as_f64_roundtrip`
- Mutability preservation:
  - `test_ptr_as_preserves_mutability_and_writes` — confirms that
    `ptr_as_i32(*mut u8)` yields a `*mut i32` whose writes mutate the
    underlying storage. Locks in the `mutable: true` propagation in
    `check_ptr_as`.
- Typeck negative tests:
  - `test_ptr_as_rejects_non_pointer_argument` — scalar input → typeck error
  - `test_ptr_as_rejects_wrong_arity` — two args → typeck error

The tests use `assert_c_interop` (the standard E2E harness) and write
through both `*const` and `*mut` views to exercise the
mutability-preserving path in `check_ptr_as` (`src/typeck/intrinsics.rs:443`).

## Gaps deferred

None. The pre-existing test surface for the branch-touched intrinsics
was already comprehensive — the only family with zero tests was
`ptr_as_*`, and that gap is now closed.

Two categories of `N/A` not flagged as gaps:

- **IR-level guards for the lane family**: the AVX-512 lane intrinsics
  (`concat_*`, `lo/hi_*`, `shuffle_i32x{8,16}`, `blend_i32`,
  `bcast_*_pairs_*`) are simple `shufflevector` / `insertelement` lowerings
  with no "silent fallback" failure mode. E2E behavior tests in
  `phase_b_avx512_lane.rs` are sufficient. Per audit plan: "skip IR
  guards for intrinsics that just add a typeck arm + dispatch arm."
- **`ptr_as_*` IR/cross-arch columns**: the codegen is literally
  `return self.compile_expr(&args[0], function)` — a no-op reinterpret
  on the same LLVM pointer value. There is no IR pattern to pin and
  the lowering is architecture-agnostic, so both columns are correctly
  N/A.

## Final test count

- Baseline (Phase 3 head, commit `3559c51`): **765 PASS / 0 FAIL**
- After Phase 4 (`+13` from `tests/ptr_as_tests.rs`): **778 PASS / 0 FAIL**

`cargo fmt --check`: clean.
`cargo clippy --all-targets --all-features -- -D warnings`: clean.
