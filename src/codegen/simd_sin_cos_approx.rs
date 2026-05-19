//! Polynomial-based vector sin and cos for f32 vectors.
//!
//! Both intrinsics share a common range-reduction + polynomial core:
//! reduce input mod π/2 via a 3-piece Cody-Waite split, compute both sin
//! and cos polynomials over the reduced argument in `[-π/4, π/4]`, then
//! blend by quadrant. `cos_approx_f32` reuses the same machinery with
//! `q += 1` — a precision-free integer shift that expresses the
//! mathematical identity `cos(x) = sin(x + π/2)`. Avoids `@llvm.sin` /
//! `@llvm.cos`, which LLVM scalarizes to per-lane libm `sinf` / `cosf`
//! on every supported architecture.
//!
//! Defined input range: [-1e7, 1e7] radians. Beyond this, the 3-piece
//! Cody-Waite split loses bits in the `q · π/2` subtraction and the
//! reduced argument's precision degrades. Maximum absolute error within
//! the defined range: ~3e-6.

use inkwell::IntPredicate;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, VectorValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

// 2/π — input multiplier for quadrant rounding.
const TWO_OVER_PI: f32 = std::f32::consts::FRAC_2_PI;

// 2-piece Cody-Waite split of π/2. PI_2_HI is the f32 closest to π/2
// (slightly larger than the true value); PI_2_LO is the negative
// residual. Sum reproduces π/2 to within ~2e-15 in f64. With FMA in
// the codegen, the effective reduction precision is ~47 bits — enough
// for inputs up to ~1e7 radians, our documented range.
//
// Earlier drafts tried a 3-piece Sleef-style split derived by halving
// Sleef's PI_A_F/B_F/C_F constants, but the values I wrote down didn't
// actually sum to π/2 (off by ~6e-8). The 2-piece Eigen-style split
// here is precision-exact and simpler.
#[allow(clippy::excessive_precision)]
const PI_2_HI: f32 = 1.5707963705062866;
#[allow(clippy::excessive_precision)]
const PI_2_LO: f32 = -4.3711388e-8;

// Polynomial coefficients for the reduced argument d' ∈ [-π/4, π/4].
//
// sin(d') ≈ d' + d'³·(-1/6) + d'⁵·(1/120) + d'⁷·(-1/5040)
//        = d' · (1 + s·(SIN_C1 + s·(SIN_C2 + s·SIN_C3)))     where s = d'²
//
// cos(d') ≈ 1 + d'²·(-1/2) + d'⁴·(1/24) + d'⁶·(-1/720) + d'⁸·(1/40320)
//        = Horner(COS_C0, COS_C1, COS_C2, COS_C3, COS_C4; s) — degree 4 in s
//
// Truncation error within d' ∈ [-π/4, π/4]:
//   sin: |d'|⁹/9! ≤ (π/4)⁹/362880 ≈ 3.4e-7  → well within 3e-6 target
//   cos: |d'|¹⁰/10! ≤ (π/4)¹⁰/3628800 ≈ 2.6e-8 → effectively zero
//
// Coefficients are Taylor values (well within the 3e-6 target without
// extra minimax tuning at this range).
#[allow(clippy::excessive_precision)]
mod coeffs {
    pub(super) const SIN_C1: f32 = -1.666_666_7e-1; // -1/6
    pub(super) const SIN_C2: f32 = 8.333_333_3e-3; // 1/120
    pub(super) const SIN_C3: f32 = -1.984_127_0e-4; // -1/5040

    pub(super) const COS_C0: f32 = 1.0;
    pub(super) const COS_C1: f32 = -5.0e-1; // -1/2
    pub(super) const COS_C2: f32 = 4.166_666_7e-2; // 1/24
    pub(super) const COS_C3: f32 = -1.388_888_9e-3; // -1/720
    pub(super) const COS_C4: f32 = 2.480_158_7e-5; // 1/40320
}
use coeffs::*;

impl<'ctx> CodeGenerator<'ctx> {
    /// Compile `sin_approx_f32(v: f32xN) -> f32xN`.
    pub(super) fn compile_sin_approx_f32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_sin_cos_core(args, function, false)
    }

    /// Compile `cos_approx_f32(v: f32xN) -> f32xN`.
    pub(super) fn compile_cos_approx_f32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_sin_cos_core(args, function, true)
    }

    /// Shared core. `want_cos = true` adds 1 to the quadrant index `q`
    /// before the blend — mathematically equivalent to `cos(x) = sin(x + π/2)`
    /// but precision-free (integer shift) rather than loss-prone (adding
    /// π/2 to large floating-point inputs).
    fn compile_sin_cos_core(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        want_cos: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        let v = match val {
            BasicValueEnum::VectorValue(vv) => vv,
            _ => {
                let name = if want_cos {
                    "cos_approx_f32"
                } else {
                    "sin_approx_f32"
                };
                return Err(CompileError::codegen_error(format!(
                    "{name} expects f32 vector, got scalar; use {} for scalar libm-precision",
                    if want_cos { "cos()" } else { "sin()" }
                )));
            }
        };
        let vec_ty = v.get_type();
        let elem_ty = vec_ty.get_element_type();
        if !elem_ty.is_float_type() || elem_ty.into_float_type() != self.context.f32_type() {
            let name = if want_cos {
                "cos_approx_f32"
            } else {
                "sin_approx_f32"
            };
            return Err(CompileError::codegen_error(format!(
                "{name} expects f32 element type"
            )));
        }
        let width = vec_ty.get_size();
        let i32_vec_ty = self.context.i32_type().vec_type(width);

        // 1. Quadrant: qf = nearbyint(v · 2/π); q = fptosi(qf)
        let two_over_pi = self.splat_f32_const_sc(TWO_OVER_PI, width)?;
        let v_scaled = self
            .builder
            .build_float_mul(v, two_over_pi, "sc_v_scaled")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let nearbyint_name = format!("llvm.nearbyint.v{width}f32");
        let nearbyint_fn = self
            .module
            .get_function(&nearbyint_name)
            .unwrap_or_else(|| {
                let fn_ty = vec_ty.fn_type(&[vec_ty.into()], false);
                self.module.add_function(&nearbyint_name, fn_ty, None)
            });
        let qf = self
            .builder
            .build_call(nearbyint_fn, &[v_scaled.into()], "sc_qf")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("nearbyint returned no value"))?
            .into_vector_value();
        let q = self
            .builder
            .build_float_to_signed_int(qf, i32_vec_ty, "sc_q")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 2. Cody-Waite range reduction: d' = v - qf · PI_2_HI - qf · PI_2_LO
        //    Each fma(neg_qf, c, accumulator) preserves precision; the second
        //    step adds the residual that PI_2_HI can't capture.
        let neg_qf = self
            .builder
            .build_float_neg(qf, "sc_neg_qf")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let pi_2_hi = self.splat_f32_const_sc(PI_2_HI, width)?;
        let pi_2_lo = self.splat_f32_const_sc(PI_2_LO, width)?;
        let d = self.fma_sc(neg_qf, pi_2_hi, v, "sc_d1", width)?;
        let d = self.fma_sc(neg_qf, pi_2_lo, d, "sc_d", width)?;
        // d ∈ [-π/4, π/4]

        // 3. s = d² (reused by both polynomials)
        let s = self
            .builder
            .build_float_mul(d, d, "sc_s")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 4. sin polynomial: sin(d) = d · (1 + s·(C1 + s·(C2 + s·C3)))
        //                          = d + d·s·u   where u = C1 + s·C2 + s²·C3
        let sin_c1 = self.splat_f32_const_sc(SIN_C1, width)?;
        let sin_c2 = self.splat_f32_const_sc(SIN_C2, width)?;
        let sin_c3 = self.splat_f32_const_sc(SIN_C3, width)?;
        let sin_u = self.fma_sc(sin_c3, s, sin_c2, "sc_sin_u1", width)?;
        let sin_u = self.fma_sc(sin_u, s, sin_c1, "sc_sin_u", width)?;
        let d_times_s = self
            .builder
            .build_float_mul(d, s, "sc_d_s")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let sin_val = self.fma_sc(d_times_s, sin_u, d, "sc_sin_val", width)?;

        // 5. cos polynomial: cos(d) = Horner(COS_C4, COS_C3, COS_C2, COS_C1, COS_C0; s)
        //                           = ((((C4·s + C3)·s + C2)·s + C1)·s + C0
        let cos_c0 = self.splat_f32_const_sc(COS_C0, width)?;
        let cos_c1 = self.splat_f32_const_sc(COS_C1, width)?;
        let cos_c2 = self.splat_f32_const_sc(COS_C2, width)?;
        let cos_c3 = self.splat_f32_const_sc(COS_C3, width)?;
        let cos_c4 = self.splat_f32_const_sc(COS_C4, width)?;
        let cos_val = self.fma_sc(cos_c4, s, cos_c3, "sc_cos_p1", width)?;
        let cos_val = self.fma_sc(cos_val, s, cos_c2, "sc_cos_p2", width)?;
        let cos_val = self.fma_sc(cos_val, s, cos_c1, "sc_cos_p3", width)?;
        let cos_val = self.fma_sc(cos_val, s, cos_c0, "sc_cos_val", width)?;

        // 6. Quadrant blend.
        //    For sin_approx: k = q. For cos_approx: k = q + 1 (precision-free phase shift).
        //    swap   = (k & 1) != 0   → pick cos_val instead of sin_val
        //    negate = (k & 2) != 0   → flip result sign
        let k = if want_cos {
            let one = self.splat_i32_const_sc(1, width)?;
            self.builder
                .build_int_add(q, one, "sc_k_cos")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
        } else {
            q
        };

        let mask_one = self.splat_i32_const_sc(1, width)?;
        let mask_two = self.splat_i32_const_sc(2, width)?;
        let zero = self.splat_i32_const_sc(0, width)?;
        let k_and_1 = self
            .builder
            .build_and(k, mask_one, "sc_k_and_1")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let swap_mask = self
            .builder
            .build_int_compare(IntPredicate::NE, k_and_1, zero, "sc_swap_mask")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let k_and_2 = self
            .builder
            .build_and(k, mask_two, "sc_k_and_2")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let negate_mask = self
            .builder
            .build_int_compare(IntPredicate::NE, k_and_2, zero, "sc_neg_mask")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // result = swap ? cos_val : sin_val
        let blended = self
            .builder
            .build_select(swap_mask, cos_val, sin_val, "sc_blended")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into_vector_value();
        // result = negate ? -blended : blended
        let neg_blended = self
            .builder
            .build_float_neg(blended, "sc_neg_blended")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let result = self
            .builder
            .build_select(negate_mask, neg_blended, blended, "sc_result")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(result)
    }

    fn splat_f32_const_sc(
        &self,
        value: f32,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let scalar = self.context.f32_type().const_float(value as f64);
        self.build_splat(BasicValueEnum::FloatValue(scalar), width)
    }

    fn splat_i32_const_sc(
        &self,
        value: i32,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let scalar: IntValue<'ctx> = self.context.i32_type().const_int(value as u64, true);
        self.build_splat(BasicValueEnum::IntValue(scalar), width)
    }

    fn fma_sc(
        &mut self,
        a: VectorValue<'ctx>,
        b: VectorValue<'ctx>,
        c: VectorValue<'ctx>,
        name: &str,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let vec_ty = self.context.f32_type().vec_type(width);
        let intrinsic_name = format!("llvm.fma.v{width}f32");
        let fn_ty = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into(), vec_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_ty, None));
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into(), c.into()], name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("fma returned no value"))?;
        Ok(result.into_vector_value())
    }
}
