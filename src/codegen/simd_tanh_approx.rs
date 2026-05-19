//! Polynomial-based vector tanh for f32 vectors.
//!
//! Emits a rational `P(x²) · x / Q(x²)` approximation in the
//! Eigen / TensorFlow / JAX fast-tanh family — degree-13 numerator
//! (odd in x), degree-6 denominator (even in x²), one fdiv per call.
//! Avoids `@llvm.tanh.v*f32`, which LLVM scalarizes to N sequential
//! libm `tanhf` calls on every supported architecture.
//!
//! Defined input range: all finite f32. Saturation handled by clamping
//! to [-9, 9] before the rational form (true tanhf saturates to ±1 in
//! f32 beyond ~8.32). Maximum absolute error: ~3e-7 across the body,
//! with ~5e-6 headroom in the test tolerance for compiler reassociation
//! and the reference libm tanhf's own rounding.

use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

// Eigen MathFunctionsImpl.h coefficients (public-domain MPL2).
// Numerator P(x²) · x has odd powers x¹, x³, … x¹³; we factor x out and
// evaluate P as a degree-6 polynomial in x². Coefficients ordered low → high.
//
// Literals match Eigen's published double-precision constants verbatim so
// they can be diffed against upstream; the compiler truncates them to the
// nearest representable f32 at compile time.
#[allow(clippy::excessive_precision)]
mod coeffs {
    pub(super) const ALPHA_1: f32 = 4.89352455891786e-03;
    pub(super) const ALPHA_3: f32 = 6.37261928875436e-04;
    pub(super) const ALPHA_5: f32 = 1.48572235717979e-05;
    pub(super) const ALPHA_7: f32 = 5.12229709037114e-08;
    pub(super) const ALPHA_9: f32 = -8.60467152213735e-11;
    pub(super) const ALPHA_11: f32 = 2.00018790482477e-13;
    pub(super) const ALPHA_13: f32 = -2.76076847742355e-16;

    // Denominator Q(x²) has even powers x⁰, x², x⁴, x⁶; degree-3 in x².
    pub(super) const BETA_0: f32 = 4.89352518554385e-03;
    pub(super) const BETA_2: f32 = 2.26843463243900e-03;
    pub(super) const BETA_4: f32 = 1.18534705686654e-04;
    pub(super) const BETA_6: f32 = 1.19825839466702e-06;
}
use coeffs::*;

// Saturation clamp. Beyond |x| ≈ 8.32, tanhf returns exactly ±1 in f32.
// We clamp at 9 to give the rational form a small margin where it's still
// well-conditioned, and rely on it to evaluate to ±1 within ULP at the edges.
const TANH_CLAMP: f32 = 9.0;

impl<'ctx> CodeGenerator<'ctx> {
    /// Compile `tanh_approx_f32(v: f32xN) -> f32xN`. Width inferred from operand.
    /// Emits a rational polynomial approximation directly — never calls @llvm.tanh.
    pub(super) fn compile_tanh_approx_f32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        let v = match val {
            BasicValueEnum::VectorValue(vv) => vv,
            _ => {
                return Err(CompileError::codegen_error(
                    "tanh_approx_f32 expects f32 vector, got scalar; use tanh() for scalar libm-precision",
                ));
            }
        };
        let vec_ty = v.get_type();
        let elem_ty = vec_ty.get_element_type();
        if !elem_ty.is_float_type() || elem_ty.into_float_type() != self.context.f32_type() {
            return Err(CompileError::codegen_error(
                "tanh_approx_f32 expects f32 element type",
            ));
        }
        let width = vec_ty.get_size();

        // Clamp x to [-TANH_CLAMP, TANH_CLAMP].
        let pos_clamp = self.splat_f32_const_ta(TANH_CLAMP, width)?;
        let neg_clamp = self.splat_f32_const_ta(-TANH_CLAMP, width)?;
        let x_min = self.minnum_ta(v, pos_clamp, "tanh_min", width)?;
        let x = self.maxnum_ta(x_min, neg_clamp, "tanh_clamped", width)?;

        // x² for the rational form
        let x2 = self
            .builder
            .build_float_mul(x, x, "x2")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Numerator: x · Horner(α₁, α₃, …, α₁₃ in x²)
        //   p = (((((α₁₃·x² + α₁₁)·x² + α₉)·x² + α₇)·x² + α₅)·x² + α₃)·x² + α₁
        //   num = x · p
        let a13 = self.splat_f32_const_ta(ALPHA_13, width)?;
        let a11 = self.splat_f32_const_ta(ALPHA_11, width)?;
        let a9 = self.splat_f32_const_ta(ALPHA_9, width)?;
        let a7 = self.splat_f32_const_ta(ALPHA_7, width)?;
        let a5 = self.splat_f32_const_ta(ALPHA_5, width)?;
        let a3 = self.splat_f32_const_ta(ALPHA_3, width)?;
        let a1 = self.splat_f32_const_ta(ALPHA_1, width)?;

        let p = self.fma_ta(a13, x2, a11, "ta_p1", width)?;
        let p = self.fma_ta(p, x2, a9, "ta_p2", width)?;
        let p = self.fma_ta(p, x2, a7, "ta_p3", width)?;
        let p = self.fma_ta(p, x2, a5, "ta_p4", width)?;
        let p = self.fma_ta(p, x2, a3, "ta_p5", width)?;
        let p = self.fma_ta(p, x2, a1, "ta_p6", width)?;

        let num = self
            .builder
            .build_float_mul(x, p, "ta_num")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Denominator: Horner(β₀, β₂, β₄, β₆ in x²)
        //   q = ((β₆·x² + β₄)·x² + β₂)·x² + β₀
        let b6 = self.splat_f32_const_ta(BETA_6, width)?;
        let b4 = self.splat_f32_const_ta(BETA_4, width)?;
        let b2 = self.splat_f32_const_ta(BETA_2, width)?;
        let b0 = self.splat_f32_const_ta(BETA_0, width)?;

        let q = self.fma_ta(b6, x2, b4, "ta_q1", width)?;
        let q = self.fma_ta(q, x2, b2, "ta_q2", width)?;
        let q = self.fma_ta(q, x2, b0, "ta_q3", width)?;

        let result = self
            .builder
            .build_float_div(num, q, "ta_result")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(result.into())
    }

    fn splat_f32_const_ta(
        &self,
        value: f32,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let scalar = self.context.f32_type().const_float(value as f64);
        self.build_splat(BasicValueEnum::FloatValue(scalar), width)
    }

    fn fma_ta(
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

    fn minnum_ta(
        &mut self,
        a: VectorValue<'ctx>,
        b: VectorValue<'ctx>,
        name: &str,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        self.binary_v8f32_intrinsic_ta("llvm.minnum", a, b, name, width)
    }

    fn maxnum_ta(
        &mut self,
        a: VectorValue<'ctx>,
        b: VectorValue<'ctx>,
        name: &str,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        self.binary_v8f32_intrinsic_ta("llvm.maxnum", a, b, name, width)
    }

    fn binary_v8f32_intrinsic_ta(
        &mut self,
        base: &str,
        a: VectorValue<'ctx>,
        b: VectorValue<'ctx>,
        name: &str,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let vec_ty = self.context.f32_type().vec_type(width);
        let intrinsic_name = format!("{base}.v{width}f32");
        let fn_ty = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_ty, None));
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error(format!("{base} returned no value")))?;
        Ok(result.into_vector_value())
    }
}
