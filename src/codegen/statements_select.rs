use inkwell::values::FunctionValue;

use crate::ast::{Expr, Stmt};
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Try to lower a simple if/else into a `select` instruction.
    /// Returns `Some(terminated)` if successful, `None` to fall through
    /// to the standard branch-based codegen.
    ///
    /// Handles two patterns:
    ///   if cond { x = e1 } else { x = e2 }  →  x = select(cond, e1, e2)
    ///   if cond { return e1 } else { return e2 }  →  return select(cond, e1, e2)
    pub(super) fn try_compile_select(
        &mut self,
        condition: &Expr,
        then_body: &[Stmt],
        else_body: &Option<Vec<Stmt>>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<Option<bool>> {
        let else_stmts = match else_body {
            Some(stmts) if stmts.len() == 1 => stmts,
            _ => return Ok(None),
        };
        if then_body.len() != 1 {
            return Ok(None);
        }

        match (&then_body[0], &else_stmts[0]) {
            // Pattern: if cond { x = e1 } else { x = e2 }
            (
                Stmt::Assign {
                    target: t1,
                    value: v1,
                    ..
                },
                Stmt::Assign {
                    target: t2,
                    value: v2,
                    ..
                },
            ) if t1 == t2 && Self::is_pure_expr(v1) && Self::is_pure_expr(v2) => {
                let cond = self.compile_expr(condition, function)?.into_int_value();
                let (ptr, var_type) = self.variables.get(t1).cloned().ok_or_else(|| {
                    CompileError::codegen_error(format!("undefined variable '{t1}'"))
                })?;
                let then_val = self.compile_expr_typed(v1, Some(&var_type), function)?;
                let else_val = self.compile_expr_typed(v2, Some(&var_type), function)?;
                let selected = self
                    .builder
                    .build_select(cond, then_val, else_val, "sel")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.builder
                    .build_store(ptr, selected)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(Some(false))
            }
            // Pattern: if cond { return e1 } else { return e2 }
            (Stmt::Return(Some(v1), _), Stmt::Return(Some(v2), _))
                if Self::is_pure_expr(v1) && Self::is_pure_expr(v2) =>
            {
                let cond = self.compile_expr(condition, function)?.into_int_value();
                let ret_hint = function
                    .get_name()
                    .to_str()
                    .ok()
                    .and_then(|n| self.func_signatures.get(n))
                    .and_then(|(_, ret)| ret.clone());
                let then_val = self.compile_expr_typed(v1, ret_hint.as_ref(), function)?;
                let else_val = self.compile_expr_typed(v2, ret_hint.as_ref(), function)?;
                let selected = self
                    .builder
                    .build_select(cond, then_val, else_val, "sel")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                self.builder
                    .build_return(Some(&selected))
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(Some(true))
            }
            _ => Ok(None),
        }
    }

    /// Returns true if an expression has no side effects and is safe to
    /// evaluate speculatively (both sides of a select are always evaluated).
    fn is_pure_expr(expr: &Expr) -> bool {
        match expr {
            Expr::Literal(..) | Expr::Variable(..) => true,
            Expr::Binary(l, _, r, _) => Self::is_pure_expr(l) && Self::is_pure_expr(r),
            Expr::Not(e, _) | Expr::Negate(e, _) => Self::is_pure_expr(e),
            Expr::Index { object, index, .. } => {
                Self::is_pure_expr(object) && Self::is_pure_expr(index)
            }
            Expr::FieldAccess { object, .. } => Self::is_pure_expr(object),
            // Function calls may have side effects — not safe to speculate.
            Expr::Call { .. } => false,
            Expr::Vector { elements, .. } => elements.iter().all(Self::is_pure_expr),
            Expr::ArrayLiteral(elems, _) => elems.iter().all(Self::is_pure_expr),
            Expr::StructLiteral { fields, .. } => fields.iter().all(|(_, e)| Self::is_pure_expr(e)),
        }
    }
}
