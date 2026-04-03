use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::{self, Type};

impl TypeChecker {
    /// Returns Some(type) if the call is a known intrinsic, None if it's a user function.
    pub(super) fn check_intrinsic_call(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        match name {
            "println" => Some(self.check_println(args, locals, span)),
            "splat" => Some(self.check_splat(args, locals, type_hint, span)),
            "load" => Some(self.check_load(args, locals, type_hint, span)),
            "store" => Some(self.check_store(args, locals, span)),
            "fma" => Some(self.check_fma(args, locals, span)),
            "abs" if !self.functions.contains_key("abs") => {
                Some(self.check_abs(args, locals, span))
            }
            "sqrt" | "rsqrt" | "exp" => Some(self.check_sqrt(name, args, locals, span)),
            "to_f32" | "to_f64" | "to_i16" | "to_i32" | "to_i64" => {
                Some(self.check_conversion(name, args, locals, span))
            }
            "ptr_as_i8" | "ptr_as_u8" | "ptr_as_i16" | "ptr_as_u16" | "ptr_as_i32"
            | "ptr_as_u32" | "ptr_as_i64" | "ptr_as_u64" | "ptr_as_f32" | "ptr_as_f64" => {
                Some(self.check_ptr_as(name, args, locals, span))
            }
            "reduce_add" | "reduce_max" | "reduce_min" => {
                Some(self.check_reduction(name, args, locals, span))
            }
            "reduce_add_fast" => Some(self.check_float_reduction(name, args, locals, span)),
            "shuffle" => Some(self.check_shuffle(args, locals, span)),
            "select" => Some(self.check_select(args, locals, span)),
            "widen_i8_f32x4" | "widen_u8_f32x4" => {
                Some(self.check_widen_i8_f32(name, args, locals, span, 4))
            }
            "widen_i8_f32x8" | "widen_u8_f32x8" => {
                Some(self.check_widen_i8_f32(name, args, locals, span, 8))
            }
            "widen_i8_f32x16" | "widen_u8_f32x16" => {
                Some(self.check_widen_i8_f32(name, args, locals, span, 16))
            }
            "widen_u8_i32x4" => Some(self.check_widen_u8_i32(name, args, locals, span, 4)),
            "widen_u8_i32x8" => Some(self.check_widen_u8_i32(name, args, locals, span, 8)),
            "widen_u8_i32x16" => Some(self.check_widen_u8_i32(name, args, locals, span, 16)),
            "widen_u8_f32x4_4" | "widen_u8_f32x4_8" | "widen_u8_f32x4_12" => {
                Some(self.check_widen_i8_f32(name, args, locals, span, 4))
            }
            "widen_i8_f32x4_4" | "widen_i8_f32x4_8" | "widen_i8_f32x4_12" => {
                Some(self.check_widen_i8_f32(name, args, locals, span, 4))
            }
            "widen_u8_i32x4_4" | "widen_u8_i32x4_8" | "widen_u8_i32x4_12" => {
                Some(self.check_widen_u8_i32(name, args, locals, span, 4))
            }
            "widen_u8_u16" => Some(self.check_widen_u8_u16(args, locals, span)),
            "narrow_f32x4_i8" => Some(self.check_narrow_f32x4_i8(args, locals, span)),
            "maddubs_i16" => Some(self.check_maddubs_i16(args, locals, span)),
            "madd_i16" => Some(self.check_madd_i16(args, locals, span)),
            "hadd_i16" => Some(self.check_hadd_i16(args, locals, span)),
            "vdot_i32" => Some(self.check_vdot_i32(args, locals, span)),
            "smmla_i32" => Some(self.check_smmla_i32(args, locals, span)),
            "ummla_i32" => Some(self.check_ummla_i32(args, locals, span)),
            "usmmla_i32" => Some(self.check_usmmla_i32(args, locals, span)),
            "shuffle_bytes" => Some(self.check_shuffle_bytes(args, locals, span)),
            "cvt_f16_f32" => Some(self.check_cvt_f16_f32(args, locals, span)),
            "cvt_f32_f16" => Some(self.check_cvt_f32_f16(args, locals, span)),
            "sat_add" => Some(self.check_sat_add(args, locals, span)),
            "sat_sub" => Some(self.check_sat_sub(args, locals, span)),
            "abs_diff" => Some(self.check_abs_diff(args, locals, span)),
            "round_f32x8_i32x8" => Some(self.check_round_f32x8_i32x8(args, locals, span)),
            "pack_sat_i32x8" => Some(self.check_pack_sat_i32x8(args, locals, span)),
            "pack_sat_i16x16" => Some(self.check_pack_sat_i16x16(args, locals, span)),
            "round_f32x4_i32x4" => Some(self.check_round_f32x4_i32x4(args, locals, span)),
            "pack_sat_i32x4" => Some(self.check_pack_sat_i32x4(args, locals, span)),
            "pack_sat_i16x8" => Some(self.check_pack_sat_i16x8(args, locals, span)),
            "addp_i32" => Some(self.check_addp_i32(args, locals, span)),
            "addp_i16" => Some(self.check_addp_i16(args, locals, span)),
            "wmul_i16" => Some(self.check_wmul_i16(args, locals, span)),
            "wmul_u16" => Some(self.check_wmul_u16(args, locals, span)),
            "wmul_i32" => Some(self.check_wmul_i32(args, locals, span)),
            "wmul_u32" => Some(self.check_wmul_u32(args, locals, span)),
            "prefetch" => Some(self.check_prefetch(args, locals, span)),
            "gather" => Some(self.check_gather(args, locals, span)),
            "scatter" => Some(self.check_scatter(args, locals, span)),
            "load_masked" => Some(self.check_load_masked(args, locals, type_hint, span)),
            "stream_store" => Some(self.check_store(args, locals, span)),
            "store_masked" => Some(self.check_store_masked(args, locals, span)),
            "movemask" => Some(self.check_movemask(args, locals, span)),
            "min" | "max" => Some(self.check_min_max(name, args, locals, span)),
            _ if types::parse_typed_load(name).is_some() => {
                let vec_type = types::parse_typed_load(name).unwrap();
                Some(self.check_load(args, locals, Some(&vec_type), span))
            }
            _ => None,
        }
    }

    fn check_println(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "println expects exactly 1 argument",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        if !arg_type.is_numeric() && arg_type != Type::String && !arg_type.is_vector() {
            return Err(CompileError::type_error(
                format!("println expects numeric, string, or vector argument, got {arg_type}"),
                args[0].span().clone(),
            ));
        }
        Ok(Type::Void)
    }

    fn check_splat(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "splat expects 1 argument",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        let (width, hint_elem) = match type_hint {
            Some(Type::Vector { width, elem }) => (*width, Some(elem.as_ref())),
            _ => (4, None),
        };
        match arg_type {
            Type::FloatLiteral => {
                let elem = hint_elem
                    .filter(|e| e.is_float())
                    .cloned()
                    .unwrap_or(Type::F32);
                Ok(Type::Vector {
                    elem: Box::new(elem),
                    width,
                })
            }
            Type::IntLiteral => {
                let elem = hint_elem
                    .filter(|e| e.is_integer())
                    .cloned()
                    .unwrap_or(Type::I32);
                Ok(Type::Vector {
                    elem: Box::new(elem),
                    width,
                })
            }
            concrete if concrete.is_numeric() => Ok(Type::Vector {
                elem: Box::new(concrete),
                width,
            }),
            _ => Err(CompileError::type_error(
                format!("splat expects numeric argument, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    fn check_fma(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "fma expects 3 arguments",
                span.clone(),
            ));
        }
        let t1 = self.check_expr(&args[0], locals)?;
        let t2 = self.check_expr(&args[1], locals)?;
        let t3 = self.check_expr(&args[2], locals)?;
        // Scalar float fma: accept f32, f64, or float literals
        let all_scalar_float = [&t1, &t2, &t3]
            .iter()
            .all(|t| matches!(t, Type::F32 | Type::F64 | Type::FloatLiteral));
        if all_scalar_float {
            // Resolve FloatLiteral to F32 if all are literals, otherwise use the concrete type
            let resolved = [&t1, &t2, &t3]
                .iter()
                .find(|t| matches!(t, Type::F32 | Type::F64))
                .map(|t| (*t).clone())
                .unwrap_or(Type::F32);
            return Ok(resolved);
        }
        // Vector fma
        if !t1.is_vector() || !t2.is_vector() || !t3.is_vector() {
            return Err(CompileError::type_error(
                format!("fma expects float or float vector arguments, got {t1}, {t2}, {t3}"),
                span.clone(),
            ));
        }
        types::unify_vector(&t1, &t2, span.clone())?;
        types::unify_vector(&t1, &t3, span.clone())?;
        match &t1 {
            Type::Vector { elem, .. } if !elem.is_float() => {
                return Err(CompileError::type_error(
                    "fma requires float vector arguments. fma only works on f32 or f64 vectors",
                    span.clone(),
                ));
            }
            _ => {}
        }
        Ok(t1)
    }

    fn check_sqrt(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::F32 | Type::F64 | Type::FloatLiteral => Ok(arg_type),
            Type::Vector { elem, .. } if elem.is_float() => Ok(arg_type),
            _ => Err(CompileError::type_error(
                format!("{name} expects float or float vector argument, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    fn check_prefetch(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "prefetch expects 2 arguments: (ptr, offset)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        if !matches!(ptr_type, Type::Pointer { .. }) {
            return Err(CompileError::type_error(
                format!("prefetch first argument must be a pointer, got {ptr_type}"),
                args[0].span().clone(),
            ));
        }
        let offset_type = self.check_expr(&args[1], locals)?;
        if !offset_type.is_integer() {
            return Err(CompileError::type_error(
                format!("prefetch offset must be integer, got {offset_type}"),
                args[1].span().clone(),
            ));
        }
        Ok(Type::Void)
    }

    fn check_abs(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "abs expects 1 argument",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::F32 | Type::F64 | Type::FloatLiteral => Ok(arg_type),
            Type::Vector { elem, .. } if elem.is_float() => Ok(arg_type),
            Type::Vector { elem, .. }
                if matches!(elem.as_ref(), Type::I8 | Type::I16 | Type::I32) =>
            {
                Ok(arg_type)
            }
            _ => Err(CompileError::type_error(
                format!("abs expects float or signed integer/vector argument, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    fn check_conversion(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        if !arg_type.is_numeric() {
            return Err(CompileError::type_error(
                format!("{name} expects numeric argument, got {arg_type}"),
                args[0].span().clone(),
            ));
        }
        let target = match name {
            "to_f32" => Type::F32,
            "to_f64" => Type::F64,
            "to_i16" => Type::I16,
            "to_i32" => Type::I32,
            "to_i64" => Type::I64,
            _ => unreachable!(),
        };
        Ok(target)
    }

    fn check_ptr_as(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        let (mutable, restrict) = match &arg_type {
            Type::Pointer {
                mutable, restrict, ..
            } => (*mutable, *restrict),
            _ => {
                return Err(CompileError::type_error(
                    format!("{name} expects a pointer argument, got {arg_type}"),
                    args[0].span().clone(),
                ));
            }
        };
        let inner = match name {
            "ptr_as_i8" => Type::I8,
            "ptr_as_u8" => Type::U8,
            "ptr_as_i16" => Type::I16,
            "ptr_as_u16" => Type::U16,
            "ptr_as_i32" => Type::I32,
            "ptr_as_u32" => Type::U32,
            "ptr_as_i64" => Type::I64,
            "ptr_as_u64" => Type::U64,
            "ptr_as_f32" => Type::F32,
            "ptr_as_f64" => Type::F64,
            _ => unreachable!(),
        };
        Ok(Type::Pointer {
            inner: Box::new(inner),
            mutable,
            restrict,
        })
    }
}
