use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue};

use crate::ast::{BinaryOp, Expr, Literal};
use crate::error::CompileError;
use crate::typeck::Type;
use crate::typeck::types::is_unsigned;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_expr(
        &mut self,
        expr: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_expr_typed(expr, None, function)
    }

    pub(crate) fn compile_expr_typed(
        &mut self,
        expr: &Expr,
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match expr {
            Expr::Literal(Literal::Integer(n), _) => {
                let ty = type_hint.unwrap_or(&Type::I32);
                match ty {
                    Type::I8 | Type::U8 => {
                        let val = self.context.i8_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                    Type::I16 | Type::U16 => {
                        let val = self.context.i16_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                    Type::I64 | Type::U64 => {
                        let val = self.context.i64_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                    _ => {
                        let val = self.context.i32_type().const_int(*n as u64, true);
                        Ok(BasicValueEnum::IntValue(val))
                    }
                }
            }
            Expr::Literal(Literal::Float(n), _) => {
                let ty = type_hint.unwrap_or(&Type::F64);
                match ty {
                    Type::F32 => {
                        let val = self.context.f32_type().const_float(*n);
                        Ok(BasicValueEnum::FloatValue(val))
                    }
                    _ => {
                        let val = self.context.f64_type().const_float(*n);
                        Ok(BasicValueEnum::FloatValue(val))
                    }
                }
            }
            Expr::Literal(Literal::StringLit(s), _) => {
                let global = self
                    .builder
                    .build_global_string_ptr(s, "str")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::PointerValue(global.as_pointer_value()))
            }
            Expr::Literal(Literal::Bool(b), _) => {
                let val = self.context.bool_type().const_int(*b as u64, false);
                Ok(BasicValueEnum::IntValue(val))
            }
            Expr::Not(inner, _) => {
                let val = self.compile_expr(inner, function)?;
                let int_val = val.into_int_value();
                let result = self
                    .builder
                    .build_not(int_val, "not")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            Expr::Negate(inner, _) => {
                let val = self.compile_expr_typed(inner, type_hint, function)?;
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = self
                            .builder
                            .build_int_neg(iv, "neg")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self
                            .builder
                            .build_float_neg(fv, "fneg")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::VectorValue(vv) => {
                        let is_float = vv.get_type().get_element_type().is_float_type();
                        if is_float {
                            let result = self
                                .builder
                                .build_float_neg(vv, "vneg")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::VectorValue(result))
                        } else {
                            let result = self
                                .builder
                                .build_int_neg(vv, "vneg")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::VectorValue(result))
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "unary '-' on unsupported type (internal error)",
                    )),
                }
            }
            Expr::Variable(name, _) => {
                if let Some((const_ty, const_lit)) = self.constants.get(name).cloned() {
                    return self.compile_const_literal(&const_ty, &const_lit);
                }
                let (ptr, ty) = self.variables.get(name).ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{name}'"))
                })?;
                let pointee_ty = self.llvm_type(ty);
                let val = self
                    .builder
                    .build_load(pointee_ty, *ptr, name)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(val)
            }
            Expr::Index { object, index, .. } => {
                let obj_val = self.compile_expr(object, function)?;
                let idx = self.compile_expr(index, function)?.into_int_value();
                if let BasicValueEnum::VectorValue(vec) = obj_val {
                    let val = self
                        .builder
                        .build_extract_element(vec, idx, "elem")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    Ok(val)
                } else {
                    let ptr = obj_val.into_pointer_value();
                    let inner_type = if let Expr::Variable(name, _) = object.as_ref() {
                        if let Some((_, Type::Pointer { inner, .. })) = self.variables.get(name) {
                            self.llvm_type(inner)
                        } else {
                            self.context.i32_type().into()
                        }
                    } else {
                        self.context.i32_type().into()
                    };
                    let elem_ptr =
                        unsafe { self.builder.build_gep(inner_type, ptr, &[idx], "elemptr") }
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    let val = self
                        .builder
                        .build_load(inner_type, elem_ptr, "elem")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                    Ok(val)
                }
            }
            Expr::Binary(lhs, op, rhs, _) => self.compile_binary(lhs, op, rhs, type_hint, function),
            Expr::Call { name, args, .. } => {
                if name == "println" {
                    return self.compile_println(&args[0], function);
                }
                if Self::is_simd_intrinsic(name) {
                    return self.compile_simd_call(name, args, type_hint, function);
                }

                let callee = *self.functions.get(name).ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined function '{name}'"))
                })?;

                let param_hints: Option<Vec<Type>> =
                    self.func_signatures.get(name).map(|(pts, _)| pts.clone());
                let compiled_args: Vec<BasicMetadataValueEnum> = args
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let hint = param_hints.as_ref().and_then(|pts| pts.get(i));
                        self.compile_expr_typed(a, hint, function).map(|v| v.into())
                    })
                    .collect::<Result<_, _>>()?;
                let result = self
                    .builder
                    .build_call(callee, &compiled_args, "call")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                match result.try_as_basic_value().left() {
                    Some(val) => Ok(val),
                    None => Ok(BasicValueEnum::IntValue(
                        self.context.i32_type().const_int(0, false),
                    )),
                }
            }
            Expr::Vector { elements, ty, .. } => {
                self.compile_vector_literal(elements, ty, function)
            }
            Expr::ArrayLiteral(..) => Err(CompileError::codegen_error(
                "array literals can only be used as shuffle indices",
            )),
            Expr::FieldAccess { object, field, .. } => {
                self.compile_field_access(object, field, function)
            }
            Expr::StructLiteral { name, fields, .. } => {
                self.compile_struct_literal(name, fields, function)
            }
        }
    }

    fn compile_binary(
        &mut self,
        lhs: &Expr,
        op: &BinaryOp,
        rhs: &Expr,
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match op {
            BinaryOp::And => return self.compile_short_circuit_and(lhs, rhs, function),
            BinaryOp::Or => return self.compile_short_circuit_or(lhs, rhs, function),
            _ => {}
        }

        let hint = self.infer_binary_hint(lhs, rhs, type_hint);
        let unsigned = hint.as_ref().map(is_unsigned).unwrap_or(false);
        let left = self.compile_expr_typed(lhs, hint.as_ref(), function)?;
        let right = self.compile_expr_typed(rhs, hint.as_ref(), function)?;

        if matches!(
            op,
            BinaryOp::Less
                | BinaryOp::Greater
                | BinaryOp::LessEqual
                | BinaryOp::GreaterEqual
                | BinaryOp::Equal
                | BinaryOp::NotEqual
        ) {
            return self.compile_comparison(&left, &right, op, unsigned);
        }

        match (&left, &right) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let result = match op {
                    BinaryOp::Add => self.builder.build_int_add(*l, *r, "add"),
                    BinaryOp::Subtract => self.builder.build_int_sub(*l, *r, "sub"),
                    BinaryOp::Multiply => self.builder.build_int_mul(*l, *r, "mul"),
                    BinaryOp::Divide if unsigned => {
                        self.builder.build_int_unsigned_div(*l, *r, "div")
                    }
                    BinaryOp::Divide => self.builder.build_int_signed_div(*l, *r, "div"),
                    BinaryOp::Modulo if unsigned => {
                        self.builder.build_int_unsigned_rem(*l, *r, "rem")
                    }
                    BinaryOp::Modulo => self.builder.build_int_signed_rem(*l, *r, "rem"),
                    _ => unreachable!(),
                }
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                let result = match op {
                    BinaryOp::Add => self.builder.build_float_add(*l, *r, "fadd"),
                    BinaryOp::Subtract => self.builder.build_float_sub(*l, *r, "fsub"),
                    BinaryOp::Multiply => self.builder.build_float_mul(*l, *r, "fmul"),
                    BinaryOp::Divide => self.builder.build_float_div(*l, *r, "fdiv"),
                    BinaryOp::Modulo => self.builder.build_float_rem(*l, *r, "frem"),
                    _ => unreachable!(),
                }
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::FloatValue(result))
            }
            (BasicValueEnum::VectorValue(l), BasicValueEnum::VectorValue(r)) => {
                let vec_unsigned = match hint.as_ref() {
                    Some(Type::Vector { elem, .. }) => is_unsigned(elem),
                    _ => false,
                };
                self.compile_vector_binary(*l, *r, op, vec_unsigned)
            }
            _ => Err(CompileError::codegen_error(
                "mismatched operand types in binary expression",
            )),
        }
    }

    fn infer_expr_type(&self, expr: &Expr) -> Option<Type> {
        let ty = match expr {
            Expr::Variable(name, _) => {
                if let Some((_, ty)) = self.variables.get(name) {
                    Some(ty.clone())
                } else if let Some((ty, _)) = self.constants.get(name) {
                    Some(ty.clone())
                } else {
                    None
                }
            }
            Expr::Binary(lhs, _, rhs, _) => self
                .infer_expr_type(lhs)
                .or_else(|| self.infer_expr_type(rhs)),
            Expr::Negate(inner, _) => self.infer_expr_type(inner),
            _ => None,
        };
        // Only return scalar numeric or vector types — struct types are not useful hints
        ty.filter(|t| {
            matches!(
                t,
                Type::F32
                    | Type::F64
                    | Type::I32
                    | Type::I64
                    | Type::I16
                    | Type::I8
                    | Type::U8
                    | Type::U16
                    | Type::U32
                    | Type::Vector { .. }
            )
        })
    }

    fn infer_binary_hint(&self, lhs: &Expr, rhs: &Expr, outer_hint: Option<&Type>) -> Option<Type> {
        if let Some(ty) = self.infer_expr_type(lhs) {
            return Some(ty);
        }
        if let Some(ty) = self.infer_expr_type(rhs) {
            return Some(ty);
        }
        outer_hint.cloned()
    }

    pub(super) fn arg_is_unsigned(&self, expr: &Expr) -> bool {
        if let Expr::Variable(name, _) = expr {
            if let Some((_, ty)) = self.variables.get(name) {
                return ty.is_unsigned_integer();
            }
            if let Some((ty, _)) = self.constants.get(name) {
                return ty.is_unsigned_integer();
            }
        }
        false
    }

    fn compile_const_literal(
        &self,
        ty: &Type,
        lit: &Literal,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match (ty, lit) {
            (Type::I8 | Type::U8, Literal::Integer(n)) => {
                Ok(self.context.i8_type().const_int(*n as u64, true).into())
            }
            (Type::I16 | Type::U16, Literal::Integer(n)) => {
                Ok(self.context.i16_type().const_int(*n as u64, true).into())
            }
            (Type::I32 | Type::U32, Literal::Integer(n)) => {
                Ok(self.context.i32_type().const_int(*n as u64, true).into())
            }
            (Type::I64 | Type::U64, Literal::Integer(n)) => {
                Ok(self.context.i64_type().const_int(*n as u64, true).into())
            }
            (Type::F32, Literal::Float(n)) => Ok(self.context.f32_type().const_float(*n).into()),
            (Type::F64, Literal::Float(n)) => Ok(self.context.f64_type().const_float(*n).into()),
            (Type::F32, Literal::Integer(n)) => {
                Ok(self.context.f32_type().const_float(*n as f64).into())
            }
            (Type::F64, Literal::Integer(n)) => {
                Ok(self.context.f64_type().const_float(*n as f64).into())
            }
            _ => Err(CompileError::codegen_error(
                "unsupported const type (internal error — please report this bug)",
            )),
        }
    }
}
