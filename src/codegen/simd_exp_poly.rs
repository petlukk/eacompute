//! Polynomial-based vector exp for f32 vectors.
//!
//! Emits an explicit SIMD polynomial — range reduction + degree-5 minimax +
//! ldexp via integer-bit-add to the exponent field — instead of calling
//! `@llvm.exp.v*f32` (which LLVM scalarizes to N sequential libm `expf` calls
//! on every supported architecture).
//!
//! Defined input range: [-50, 50]. Outside, output is undefined.
//! Maximum relative error: ~2^-18 (~3.8e-6) on the defined range.

use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

// Sleef-derived f32 minimax exp constants, max relative error ~2^-18.
// LOG2_E is spelled as a cast from f64 to avoid the clippy::approx_constant lint
// (the f32 value is the same bit pattern as std::f32::consts::LOG2_E).
#[allow(clippy::excessive_precision)]
const LOG2_E: f32 = std::f64::consts::LOG2_E as f32; // ≈ 1.442_695
const LN2_HI: f32 = 0.693_138_1; // Cody-Waite high word
const LN2_LO: f32 = 9.058_006e-6; // Cody-Waite low word
const POLY_C1: f32 = 1.0;
const POLY_C2: f32 = 0.5; // 0.5000000119 truncated to f32
const POLY_C3: f32 = 0.166_666_55;
const POLY_C4: f32 = 0.041_665_725;
const POLY_C5: f32 = 0.008_336_987;

impl<'ctx> CodeGenerator<'ctx> {
    /// Compile `exp_poly_f32(v: f32xN) -> f32xN`. Width inferred from operand.
    /// Emits range-reduced polynomial directly — never calls @llvm.exp.
    pub(super) fn compile_exp_poly_f32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        let v = match val {
            BasicValueEnum::VectorValue(vv) => vv,
            _ => {
                return Err(CompileError::codegen_error(
                    "exp_poly_f32 expects f32 vector, got scalar; use exp() for scalar libm-precision",
                ));
            }
        };
        let vec_ty = v.get_type();
        let elem_ty = vec_ty.get_element_type();
        if !elem_ty.is_float_type() || elem_ty.into_float_type() != self.context.f32_type() {
            return Err(CompileError::codegen_error(
                "exp_poly_f32 expects f32 element type",
            ));
        }
        let width = vec_ty.get_size();

        let i32_ty = self.context.i32_type();
        let i32_vec_ty = i32_ty.vec_type(width);

        // Splat f32 constants
        let log2_e = self.splat_f32_const_ep(LOG2_E, width)?;
        let ln2_hi = self.splat_f32_const_ep(LN2_HI, width)?;
        let ln2_lo = self.splat_f32_const_ep(LN2_LO, width)?;
        let c1 = self.splat_f32_const_ep(POLY_C1, width)?;
        let c2 = self.splat_f32_const_ep(POLY_C2, width)?;
        let c3 = self.splat_f32_const_ep(POLY_C3, width)?;
        let c4 = self.splat_f32_const_ep(POLY_C4, width)?;
        let c5 = self.splat_f32_const_ep(POLY_C5, width)?;
        let one = self.splat_f32_const_ep(1.0, width)?;

        // Step 1: fx = x * LOG2_E
        let fx = self
            .builder
            .build_float_mul(v, log2_e, "fx")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Step 2: nf = nearbyint(fx) — round to nearest, integer-valued float
        let nearbyint_name = format!("llvm.nearbyint.v{width}f32");
        let nearbyint_fn = self
            .module
            .get_function(&nearbyint_name)
            .unwrap_or_else(|| {
                let fn_ty = vec_ty.fn_type(&[vec_ty.into()], false);
                self.module.add_function(&nearbyint_name, fn_ty, None)
            });
        let nf = self
            .builder
            .build_call(nearbyint_fn, &[fx.into()], "nf")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("nearbyint returned no value"))?
            .into_vector_value();

        // Step 3: n = fptosi(nf) -> i32xN
        let n = self
            .builder
            .build_float_to_signed_int(nf, i32_vec_ty, "n")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Step 4: r = x - nf*LN2_HI - nf*LN2_LO  (Cody-Waite two-step split)
        // Uses fma(neg_nf, ln2_*, ...) for each step
        let neg_nf = self
            .builder
            .build_float_neg(nf, "neg_nf")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let r1 = self.fma_ep(neg_nf, ln2_hi, v, "r1", width)?;
        let r = self.fma_ep(neg_nf, ln2_lo, r1, "r", width)?;

        // Step 5: Horner — ((((c5*r + c4)*r + c3)*r + c2)*r + c1)*r + 1.0
        let p1 = self.fma_ep(c5, r, c4, "p1", width)?;
        let p2 = self.fma_ep(p1, r, c3, "p2", width)?;
        let p3 = self.fma_ep(p2, r, c2, "p3", width)?;
        let p4 = self.fma_ep(p3, r, c1, "p4", width)?;
        let p5 = self.fma_ep(p4, r, one, "p5", width)?;

        // Step 6: ldexp via exponent-field bit manipulation
        // result = bitcast<f32>(bitcast<i32>(p5) + (n << 23))
        let p5_bits = self
            .builder
            .build_bit_cast(p5, i32_vec_ty, "p5_bits")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into_vector_value();
        let shift_amt = self.splat_i32_const_ep(23, width)?;
        let n_shifted = self
            .builder
            .build_left_shift(n, shift_amt, "n_shl")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let result_bits = self
            .builder
            .build_int_add(p5_bits, n_shifted, "result_bits")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let result = self
            .builder
            .build_bit_cast(result_bits, vec_ty, "result")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(result)
    }

    /// Splat a constant f32 value into an f32 vector of the given width.
    fn splat_f32_const_ep(
        &self,
        value: f32,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let scalar = self.context.f32_type().const_float(value as f64);
        self.build_splat(BasicValueEnum::FloatValue(scalar), width)
    }

    /// Splat a constant i32 value into an i32 vector of the given width.
    fn splat_i32_const_ep(
        &self,
        value: i32,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let scalar = self.context.i32_type().const_int(value as u64, true);
        self.build_splat(BasicValueEnum::IntValue(scalar), width)
    }

    /// Emit `llvm.fma.v{N}f32(a, b, c)` — result = a*b + c.
    fn fma_ep(
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
