//! Polynomial-based vector natural log for f32 vectors.
//!
//! Emits an Eigen / Cephes-family log_approx: bit-level decomposition of
//! x = m · 2^e (frexp convention, m ∈ [0.5, 1)), √2/2 rebalance to center
//! the polynomial range, degree-8 Horner in (m - 1), then Cody-Waite
//! recombine with e · ln(2). Avoids `@llvm.log.v*f32`, which LLVM
//! scalarizes to N sequential libm `logf` calls on every supported
//! architecture.
//!
//! Defined input range: (0, +∞). For x ≤ 0, NaN, or ±∞, output is
//! undefined. Matches `exp_poly_f32`'s "bounded input contract" style.
//! Maximum absolute error: ~3e-6 across the defined range (compatible
//! with `exp_poly_f32`'s 2⁻¹⁸ relative target).

use inkwell::FloatPredicate;
use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

// Cody-Waite split of ln(2): LN2_HI + LN2_LO ≈ ln(2) to f32 precision.
// LN2_HI is exact-representable in f32 (0.693359375 = 22188800 / 2^25);
// LN2_LO carries the residual correction. Accumulating e · LN2_HI and
// e · LN2_LO separately, with the low part added before the dominant
// `+u` linear term, preserves precision when the integer exponent is
// large.
// LN2_HI is exact in f32 (binary `0.10110001 1`, fits in 9 mantissa bits)
// but clippy's excessive-precision lint flags any literal with this many
// digits regardless.
#[allow(clippy::excessive_precision)]
const LN2_HI: f32 = 0.693_359_375;
const LN2_LO: f32 = -2.121_944_4e-4;

// Eigen MathFunctionsImpl.h polynomial coefficients (MPL2).
// Degree-8 Horner in u = m - 1, where m ∈ [√2/2, √2) after rebalance,
// fitting the (log(1+u) - u + u²/2) / u³ tail of the Taylor expansion.
// Literals kept at Eigen's published double precision for diff-friendliness;
// the compiler truncates each to the nearest representable f32.
#[allow(clippy::excessive_precision)]
mod coeffs {
    pub(super) const P0: f32 = 7.0376836292e-2;
    pub(super) const P1: f32 = -1.1514610310e-1;
    pub(super) const P2: f32 = 1.1676998740e-1;
    pub(super) const P3: f32 = -1.2420140846e-1;
    pub(super) const P4: f32 = 1.4249322787e-1;
    pub(super) const P5: f32 = -1.6668057665e-1;
    pub(super) const P6: f32 = 2.0000714765e-1;
    pub(super) const P7: f32 = -2.4999993993e-1;
    pub(super) const P8: f32 = 3.3333331174e-1;
}
use coeffs::*;

// Frexp-convention masks. To extract m ∈ [0.5, 1):
//   m_bits = (bits & MANTISSA_MASK) | EXP_HALF_BITS
//   e_raw  = (bits >> 23) & 0xFF        (raw biased exponent)
//   e      = e_raw - 126                (so x = m · 2^e with m ∈ [0.5, 1))
const MANTISSA_MASK: i32 = 0x807F_FFFFu32 as i32; // sign + mantissa bits
const EXP_HALF_BITS: i32 = 0x3F00_0000; // exponent = 126 (i.e. 2^-1)
const EXP_BIAS: i32 = 126;
// √2/2 — rebalance boundary. Spelled via the f32 const to avoid clippy's
// approx_constant lint; the bit pattern matches the literal Eigen uses.
const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;

impl<'ctx> CodeGenerator<'ctx> {
    /// Compile `log_approx_f32(v: f32xN) -> f32xN`. Width inferred from operand.
    /// Emits the bit-decomp + polynomial + Cody-Waite recombine directly —
    /// never calls @llvm.log.
    pub(super) fn compile_log_approx_f32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        let v = match val {
            BasicValueEnum::VectorValue(vv) => vv,
            _ => {
                return Err(CompileError::codegen_error(
                    "log_approx_f32 expects f32 vector, got scalar; use log() for scalar libm-precision",
                ));
            }
        };
        let vec_ty = v.get_type();
        let elem_ty = vec_ty.get_element_type();
        if !elem_ty.is_float_type() || elem_ty.into_float_type() != self.context.f32_type() {
            return Err(CompileError::codegen_error(
                "log_approx_f32 expects f32 element type",
            ));
        }
        let width = vec_ty.get_size();
        let i32_vec_ty = self.context.i32_type().vec_type(width);

        // 1. Bitcast input to integer for bit-level extraction.
        let bits = self
            .builder
            .build_bit_cast(v, i32_vec_ty, "log_bits")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into_vector_value();

        // 2. e_raw = bits >> 23 (logical right shift — high bits become zero).
        let shift_23 = self.splat_i32_const_la(23, width)?;
        let shifted = self
            .builder
            .build_right_shift(bits, shift_23, false, "log_shr")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        // Mask off sign bit: e_raw = shifted & 0xFF.
        let mask_ff = self.splat_i32_const_la(0xFF, width)?;
        let e_raw = self
            .builder
            .build_and(shifted, mask_ff, "log_e_raw")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 3. e = e_raw - 126 (frexp bias).
        let bias = self.splat_i32_const_la(EXP_BIAS, width)?;
        let e_int = self
            .builder
            .build_int_sub(e_raw, bias, "log_e_int")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 4. Mantissa: clear original exponent bits, set them to 126 (i.e. 2^-1).
        //    m_bits = (bits & 0x807FFFFF) | 0x3F000000;  m = bitcast<f32>(m_bits).
        let mantissa_mask = self.splat_i32_const_la(MANTISSA_MASK, width)?;
        let exp_half = self.splat_i32_const_la(EXP_HALF_BITS, width)?;
        let masked = self
            .builder
            .build_and(bits, mantissa_mask, "log_mant_masked")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let m_bits = self
            .builder
            .build_or(masked, exp_half, "log_m_bits")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let m = self
            .builder
            .build_bit_cast(m_bits, vec_ty, "log_m")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into_vector_value();
        // m ∈ [0.5, 1).

        // 5. √2/2 rebalance. If m < √2/2, double m and decrement e.
        //    Result: m ∈ [√2/2, √2), u = m - 1 ∈ [-0.293, 0.414].
        let sqrt_half = self.splat_f32_const_la(SQRT_HALF, width)?;
        let m_lt = self
            .builder
            .build_float_compare(FloatPredicate::OLT, m, sqrt_half, "log_m_lt_sqrth")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let m_doubled = self
            .builder
            .build_float_add(m, m, "log_m_doubled")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let m_rebalanced = self
            .builder
            .build_select(m_lt, m_doubled, m, "log_m_rebal")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into_vector_value();
        let one_i32 = self.splat_i32_const_la(1, width)?;
        let e_decremented = self
            .builder
            .build_int_sub(e_int, one_i32, "log_e_dec")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let e_rebalanced = self
            .builder
            .build_select(m_lt, e_decremented, e_int, "log_e_rebal")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into_vector_value();

        // 6. Convert e (i32) to f32.
        let e_f32 = self
            .builder
            .build_signed_int_to_float(e_rebalanced, vec_ty, "log_e_f32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 7. u = m - 1.
        let one_f32 = self.splat_f32_const_la(1.0, width)?;
        let u = self
            .builder
            .build_float_sub(m_rebalanced, one_f32, "log_u")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 8. u² (reused in polynomial-tail correction).
        let u2 = self
            .builder
            .build_float_mul(u, u, "log_u2")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 9. Horner: P(u) = ((((((((p0·u + p1)·u + p2)·u + p3)·u + p4)·u + p5)·u + p6)·u + p7)·u + p8.
        let p0 = self.splat_f32_const_la(P0, width)?;
        let p1 = self.splat_f32_const_la(P1, width)?;
        let p2 = self.splat_f32_const_la(P2, width)?;
        let p3 = self.splat_f32_const_la(P3, width)?;
        let p4 = self.splat_f32_const_la(P4, width)?;
        let p5 = self.splat_f32_const_la(P5, width)?;
        let p6 = self.splat_f32_const_la(P6, width)?;
        let p7 = self.splat_f32_const_la(P7, width)?;
        let p8 = self.splat_f32_const_la(P8, width)?;

        let poly = self.fma_la(p0, u, p1, "log_poly1", width)?;
        let poly = self.fma_la(poly, u, p2, "log_poly2", width)?;
        let poly = self.fma_la(poly, u, p3, "log_poly3", width)?;
        let poly = self.fma_la(poly, u, p4, "log_poly4", width)?;
        let poly = self.fma_la(poly, u, p5, "log_poly5", width)?;
        let poly = self.fma_la(poly, u, p6, "log_poly6", width)?;
        let poly = self.fma_la(poly, u, p7, "log_poly7", width)?;
        let poly = self.fma_la(poly, u, p8, "log_poly8", width)?;
        // Now poly = P(u), degree 8.

        // 10. y = P(u) · u · u² = P(u) · u³.
        let y = self
            .builder
            .build_float_mul(poly, u, "log_pu")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let y = self
            .builder
            .build_float_mul(y, u2, "log_y_corr")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // 11. Cody-Waite recombine — small-magnitude terms first.
        //     y = y - 0.5·u²    (the -u²/2 term of log(1+u) Taylor)
        let neg_half = self.splat_f32_const_la(-0.5, width)?;
        let y = self.fma_la(u2, neg_half, y, "log_y_minus_half_u2", width)?;

        //     y = y + e_f32 · LN2_LO    (low part of ln(2)·e)
        let ln2_lo = self.splat_f32_const_la(LN2_LO, width)?;
        let y = self.fma_la(e_f32, ln2_lo, y, "log_y_plus_e_lo", width)?;

        //     y = y + u    (linear term — dominant for small u)
        let y = self
            .builder
            .build_float_add(y, u, "log_y_plus_u")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        //     y = y + e_f32 · LN2_HI    (dominant ln(2)·e contribution last)
        let ln2_hi = self.splat_f32_const_la(LN2_HI, width)?;
        let y = self.fma_la(e_f32, ln2_hi, y, "log_y_plus_e_hi", width)?;

        Ok(y.into())
    }

    fn splat_f32_const_la(
        &self,
        value: f32,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let scalar = self.context.f32_type().const_float(value as f64);
        self.build_splat(BasicValueEnum::FloatValue(scalar), width)
    }

    fn splat_i32_const_la(
        &self,
        value: i32,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let scalar = self.context.i32_type().const_int(value as u64, true);
        self.build_splat(BasicValueEnum::IntValue(scalar), width)
    }

    fn fma_la(
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
