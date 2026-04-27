use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;
use crate::typeck::types as typeck_types;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Returns true if the argument expression is a vector variable with an unsigned element type.
    pub(crate) fn infer_unsigned_elem_from_arg(&self, arg: &Expr) -> bool {
        if let Expr::Variable(name, _) = arg
            && let Some((_, Type::Vector { elem, .. })) = self.variables.get(name)
        {
            return matches!(elem.as_ref(), Type::U8 | Type::U16);
        }
        false
    }

    pub(crate) fn is_simd_intrinsic(name: &str) -> bool {
        matches!(
            name,
            "splat"
                | "load"
                | "store"
                | "stream_store"
                | "gather"
                | "scatter"
                | "load_masked"
                | "store_masked"
                | "fma"
                | "abs"
                | "sqrt"
                | "rsqrt"
                | "exp"
                | "reduce_add"
                | "reduce_add_fast"
                | "reduce_max"
                | "reduce_min"
                | "shuffle"
                | "select"
                | "widen_i8_f32x4"
                | "widen_u8_f32x4"
                | "widen_i8_f32x8"
                | "widen_u8_f32x8"
                | "widen_i8_f32x16"
                | "widen_u8_f32x16"
                | "widen_u8_i32x4"
                | "widen_u8_i32x8"
                | "widen_u8_i32x16"
                | "widen_u8_f32x4_4"
                | "widen_u8_f32x4_8"
                | "widen_u8_f32x4_12"
                | "widen_i8_f32x4_4"
                | "widen_i8_f32x4_8"
                | "widen_i8_f32x4_12"
                | "widen_u8_i32x4_4"
                | "widen_u8_i32x4_8"
                | "widen_u8_i32x4_12"
                | "narrow_f32x4_i8"
                | "maddubs_i16"
                | "madd_i16"
                | "hadd_i16"
                | "vdot_i32"
                | "vdot_lane_i32"
                | "smmla_i32"
                | "ummla_i32"
                | "usmmla_i32"
                | "shuffle_bytes"
                | "to_f32"
                | "to_f64"
                | "to_i16"
                | "to_i32"
                | "to_i64"
                | "prefetch"
                | "movemask"
                | "min"
                | "max"
                | "sat_add"
                | "sat_sub"
                | "abs_diff"
                | "addp_i32"
                | "addp_i16"
                | "wmul_i16"
                | "wmul_u16"
                | "wmul_i32"
                | "wmul_u32"
                | "round_f32x8_i32x8"
                | "pack_sat_i32x8"
                | "pack_sat_i16x16"
                | "round_f32x4_i32x4"
                | "pack_sat_i32x4"
                | "pack_sat_i16x8"
                | "pack_usat_i32x8"
                | "pack_usat_i16x16"
                | "pack_usat_i32x4"
                | "pack_usat_i16x8"
                | "bsrli_i8x16"
                | "bsrli_i8x32"
                | "bslli_i8x16"
                | "bslli_i8x32"
                | "cvt_f16_f32"
                | "cvt_f32_f16"
                | "widen_u8_u16"
                | "concat_i8x16"
                | "concat_u8x16"
                | "concat_i8x32"
                | "concat_u8x32"
                | "concat_i32x8"
                | "concat_f32x8"
                | "lo128_i8x32"
                | "hi128_i8x32"
                | "lo128_u8x32"
                | "hi128_u8x32"
                | "lo256_i8x64"
                | "hi256_i8x64"
                | "lo256_u8x64"
                | "hi256_u8x64"
                | "lo256_i32x16"
                | "hi256_i32x16"
                | "lo256_f32x16"
                | "hi256_f32x16"
                | "bcast_even_pairs_i32x8"
                | "bcast_odd_pairs_i32x8"
                | "bcast_even_pairs_i32x16"
                | "bcast_odd_pairs_i32x16"
                | "shuffle_i32x8"
                | "shuffle_i32x16"
                | "blend_i32"
                | "bitcast_i8x16"
                | "bitcast_i8x32"
                | "bitcast_i32x4"
                | "bitcast_i32x8"
                | "f32x4_from_scalars"
                | "f32x8_from_scalars"
        ) || typeck_types::parse_typed_load(name).is_some()
    }

    /// Returns true if any arg is an f16 vector variable, or if the type_hint is an f16 vector.
    fn call_uses_f16(&self, args: &[Expr], type_hint: Option<&Type>) -> bool {
        if let Some(Type::Vector { elem, .. }) = type_hint
            && matches!(**elem, Type::F16)
        {
            return true;
        }
        for a in args {
            if let Expr::Variable(name, _) = a
                && let Some((_, Type::Vector { elem, .. })) = self.variables.get(name)
                && matches!(**elem, Type::F16)
            {
                return true;
            }
        }
        false
    }

    pub(crate) fn compile_simd_call(
        &mut self,
        name: &str,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.fp16 && self.call_uses_f16(args, type_hint) {
            return Err(CompileError::codegen_error(
                "f16 vector types require --fp16; use cvt_f16_f32 to compute through f32 instead",
            ));
        }
        match name {
            "splat" => {
                let elem_hint = if let Some(Type::Vector { elem, .. }) = type_hint {
                    Some(elem.as_ref())
                } else {
                    None
                };
                if matches!(elem_hint, Some(Type::F16)) {
                    return self.compile_splat_f16(args, type_hint, function);
                }
                let width = match type_hint {
                    Some(Type::Vector { width, .. }) => *width as u32,
                    _ => 4,
                };
                let val = self.compile_expr_typed(&args[0], elem_hint, function)?;
                let vec = self.build_splat(val, width)?;
                Ok(BasicValueEnum::VectorValue(vec))
            }
            "load" => self.compile_load(args, type_hint, function),
            "store" => self.compile_store(args, function),
            "stream_store" => self.compile_stream_store(args, function),
            "gather" => self.compile_gather(args, type_hint, function),
            "scatter" => self.compile_scatter(args, function),
            "load_masked" => self.compile_load_masked(args, type_hint, function),
            "store_masked" => self.compile_store_masked(args, function),
            "fma" => self.compile_fma(args, function),
            "abs" => self.compile_abs(args, function),
            "sqrt" => self.compile_sqrt(args, function),
            "rsqrt" => self.compile_rsqrt(args, function),
            "exp" => self.compile_exp(args, function),
            "reduce_add" | "reduce_add_fast" | "reduce_max" | "reduce_min" => {
                self.compile_reduce(args, name, function)
            }
            "shuffle" => self.compile_shuffle(args, function),
            "select" => self.compile_select(args, function),
            "widen_i8_f32x4" => self.compile_widen_i8_f32(args, false, 4, 0, function),
            "widen_u8_f32x4" => self.compile_widen_i8_f32(args, true, 4, 0, function),
            "widen_i8_f32x8" => self.compile_widen_i8_f32(args, false, 8, 0, function),
            "widen_u8_f32x8" => self.compile_widen_i8_f32(args, true, 8, 0, function),
            "widen_i8_f32x16" => self.compile_widen_i8_f32(args, false, 16, 0, function),
            "widen_u8_f32x16" => self.compile_widen_i8_f32(args, true, 16, 0, function),
            "widen_u8_i32x4" => self.compile_widen_u8_i32(args, 4, 0, function),
            "widen_u8_i32x8" => self.compile_widen_u8_i32(args, 8, 0, function),
            "widen_u8_i32x16" => self.compile_widen_u8_i32(args, 16, 0, function),
            "widen_u8_f32x4_4" => self.compile_widen_i8_f32(args, true, 4, 4, function),
            "widen_u8_f32x4_8" => self.compile_widen_i8_f32(args, true, 4, 8, function),
            "widen_u8_f32x4_12" => self.compile_widen_i8_f32(args, true, 4, 12, function),
            "widen_i8_f32x4_4" => self.compile_widen_i8_f32(args, false, 4, 4, function),
            "widen_i8_f32x4_8" => self.compile_widen_i8_f32(args, false, 4, 8, function),
            "widen_i8_f32x4_12" => self.compile_widen_i8_f32(args, false, 4, 12, function),
            "widen_u8_i32x4_4" => self.compile_widen_u8_i32(args, 4, 4, function),
            "widen_u8_i32x4_8" => self.compile_widen_u8_i32(args, 4, 8, function),
            "widen_u8_i32x4_12" => self.compile_widen_u8_i32(args, 4, 12, function),
            "narrow_f32x4_i8" => self.compile_narrow_f32x4_i8(args, function),
            "maddubs_i16" => self.compile_maddubs_i16(args, function),
            "madd_i16" => self.compile_madd_i16(args, function),
            "hadd_i16" => self.compile_hadd_i16(args, function),
            "vdot_i32" => self.compile_vdot_i32(args, function),
            "vdot_lane_i32" => {
                let lane = Self::extract_imm8(&args[3])?;
                self.compile_vdot_lane_i32(args, function, lane)
            }
            "smmla_i32" => self.compile_smmla_i32(args, function),
            "ummla_i32" => self.compile_ummla_i32(args, function),
            "usmmla_i32" => self.compile_usmmla_i32(args, function),
            "shuffle_bytes" => self.compile_shuffle_bytes(args, function),
            "to_f32" | "to_f64" | "to_i16" | "to_i32" | "to_i64" => {
                self.compile_conversion(name, args, function)
            }
            "prefetch" => self.compile_prefetch(args, function),
            "movemask" => self.compile_movemask(args, function),
            "min" | "max" => self.compile_min_max(args, name, function),
            "sat_add" => self.compile_sat_add(args, function),
            "sat_sub" => self.compile_sat_sub(args, function),
            "abs_diff" => self.compile_abs_diff(args, function),
            "addp_i32" => self.compile_addp_i32(args, function),
            "addp_i16" => self.compile_addp_i16(args, function),
            "wmul_i16" => self.compile_wmul_i16(args, function),
            "wmul_u16" => self.compile_wmul_u16(args, function),
            "wmul_i32" => self.compile_wmul_i32(args, function),
            "wmul_u32" => self.compile_wmul_u32(args, function),
            "round_f32x8_i32x8" => self.compile_round_f32x8_i32x8(args, function),
            "pack_sat_i32x8" => self.compile_pack_sat_i32x8(args, function),
            "pack_sat_i16x16" => self.compile_pack_sat_i16x16(args, function),
            "round_f32x4_i32x4" => self.compile_round_f32x4_i32x4(args, function),
            "pack_sat_i32x4" => self.compile_pack_sat_i32x4(args, function),
            "pack_sat_i16x8" => self.compile_pack_sat_i16x8(args, function),
            "pack_usat_i32x8" => self.compile_pack_usat_i32x8(args, function),
            "pack_usat_i16x16" => self.compile_pack_usat_i16x16(args, function),
            "pack_usat_i32x4" => self.compile_pack_usat_i32x4(args, function),
            "pack_usat_i16x8" => self.compile_pack_usat_i16x8(args, function),
            "bsrli_i8x16" => {
                let imm = Self::extract_imm8(&args[1])?;
                self.compile_bsrli_i8x16(args, function, imm)
            }
            "bsrli_i8x32" => {
                let imm = Self::extract_imm8(&args[1])?;
                self.compile_bsrli_i8x32(args, function, imm)
            }
            "bslli_i8x16" => {
                let imm = Self::extract_imm8(&args[1])?;
                self.compile_bslli_i8x16(args, function, imm)
            }
            "bslli_i8x32" => {
                let imm = Self::extract_imm8(&args[1])?;
                self.compile_bslli_i8x32(args, function, imm)
            }
            "cvt_f16_f32" => self.compile_cvt_f16_f32(args, function),
            "cvt_f32_f16" => self.compile_cvt_f32_f16(args, function),
            "bitcast_i8x16" => self.compile_bitcast(args, function, Type::I8, 16),
            "bitcast_i8x32" => self.compile_bitcast(args, function, Type::I8, 32),
            "bitcast_i32x4" => self.compile_bitcast(args, function, Type::I32, 4),
            "bitcast_i32x8" => self.compile_bitcast(args, function, Type::I32, 8),
            "widen_u8_u16" => self.compile_widen_u8_u16(args, function),
            "concat_i8x16" | "concat_u8x16" | "concat_i8x32" | "concat_u8x32" | "concat_i32x8"
            | "concat_f32x8" => self.emit_concat(args, function),
            "lo128_i8x32" | "lo128_u8x32" | "lo256_i8x64" | "lo256_u8x64" | "lo256_i32x16"
            | "lo256_f32x16" => self.emit_lo_extract(args, function),
            "hi128_i8x32" | "hi128_u8x32" | "hi256_i8x64" | "hi256_u8x64" | "hi256_i32x16"
            | "hi256_f32x16" => self.emit_hi_extract(args, function),
            "shuffle_i32x8" | "shuffle_i32x16" => {
                let imm = Self::extract_imm8(&args[1])?;
                self.emit_shuffle_i32(args, function, imm)
            }
            "blend_i32" => {
                let imm = Self::extract_imm8(&args[2])?;
                self.emit_blend_i32(args, function, imm)
            }
            "bcast_even_pairs_i32x8" | "bcast_even_pairs_i32x16" => {
                self.emit_bcast_pairs(args, function, false)
            }
            "bcast_odd_pairs_i32x8" | "bcast_odd_pairs_i32x16" => {
                self.emit_bcast_pairs(args, function, true)
            }
            "f32x4_from_scalars" => self.emit_f32_from_scalars(args, function, 4),
            "f32x8_from_scalars" => self.emit_f32_from_scalars(args, function, 8),
            _ if typeck_types::parse_typed_load(name).is_some() => {
                let vec_type = typeck_types::parse_typed_load(name).unwrap();
                self.compile_load(args, Some(&vec_type), function)
            }
            _ => Err(CompileError::codegen_error(format!(
                "unknown SIMD intrinsic '{name}'"
            ))),
        }
    }
}
