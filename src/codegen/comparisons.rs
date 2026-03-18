use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate};

use crate::ast::{BinaryOp, Expr};
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(super) fn compile_comparison(
        &mut self,
        left: &BasicValueEnum<'ctx>,
        right: &BasicValueEnum<'ctx>,
        op: &BinaryOp,
        unsigned: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match (left, right) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let pred = match op {
                    BinaryOp::Less => {
                        if unsigned {
                            IntPredicate::ULT
                        } else {
                            IntPredicate::SLT
                        }
                    }
                    BinaryOp::Greater => {
                        if unsigned {
                            IntPredicate::UGT
                        } else {
                            IntPredicate::SGT
                        }
                    }
                    BinaryOp::LessEqual => {
                        if unsigned {
                            IntPredicate::ULE
                        } else {
                            IntPredicate::SLE
                        }
                    }
                    BinaryOp::GreaterEqual => {
                        if unsigned {
                            IntPredicate::UGE
                        } else {
                            IntPredicate::SGE
                        }
                    }
                    BinaryOp::Equal => IntPredicate::EQ,
                    BinaryOp::NotEqual => IntPredicate::NE,
                    _ => unreachable!(),
                };
                let result = self
                    .builder
                    .build_int_compare(pred, *l, *r, "cmp")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                let pred = match op {
                    BinaryOp::Less => FloatPredicate::OLT,
                    BinaryOp::Greater => FloatPredicate::OGT,
                    BinaryOp::LessEqual => FloatPredicate::OLE,
                    BinaryOp::GreaterEqual => FloatPredicate::OGE,
                    BinaryOp::Equal => FloatPredicate::OEQ,
                    BinaryOp::NotEqual => FloatPredicate::ONE,
                    _ => unreachable!(),
                };
                let result = self
                    .builder
                    .build_float_compare(pred, *l, *r, "fcmp")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::IntValue(result))
            }
            _ => Err(CompileError::codegen_error(
                "mismatched types in comparison",
            )),
        }
    }

    /// Short-circuit AND: if left is false, result is false (skip right)
    pub(super) fn compile_short_circuit_and(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let left = self.compile_expr(lhs, function)?.into_int_value();
        let lhs_bb = self.builder.get_insert_block().unwrap();

        let rhs_bb = self.context.append_basic_block(function, "and_rhs");
        let merge_bb = self.context.append_basic_block(function, "and_merge");

        self.builder
            .build_conditional_branch(left, rhs_bb, merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(rhs_bb);
        let right = self.compile_expr(rhs, function)?.into_int_value();
        let rhs_end_bb = self.builder.get_insert_block().unwrap();
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(merge_bb);
        let bool_ty = self.context.bool_type();
        let phi = self
            .builder
            .build_phi(bool_ty, "and_result")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let false_val = bool_ty.const_int(0, false);
        phi.add_incoming(&[(&false_val, lhs_bb), (&right, rhs_end_bb)]);

        Ok(BasicValueEnum::IntValue(
            phi.as_basic_value().into_int_value(),
        ))
    }

    /// Short-circuit OR: if left is true, result is true (skip right)
    pub(super) fn compile_short_circuit_or(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let left = self.compile_expr(lhs, function)?.into_int_value();
        let lhs_bb = self.builder.get_insert_block().unwrap();

        let rhs_bb = self.context.append_basic_block(function, "or_rhs");
        let merge_bb = self.context.append_basic_block(function, "or_merge");

        self.builder
            .build_conditional_branch(left, merge_bb, rhs_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(rhs_bb);
        let right = self.compile_expr(rhs, function)?.into_int_value();
        let rhs_end_bb = self.builder.get_insert_block().unwrap();
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        self.builder.position_at_end(merge_bb);
        let bool_ty = self.context.bool_type();
        let phi = self
            .builder
            .build_phi(bool_ty, "or_result")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let true_val = bool_ty.const_int(1, false);
        phi.add_incoming(&[(&true_val, lhs_bb), (&right, rhs_end_bb)]);

        Ok(BasicValueEnum::IntValue(
            phi.as_basic_value().into_int_value(),
        ))
    }
}
