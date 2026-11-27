use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

/// The lane type and count a typed masked load/store names, if it is one.
pub(crate) fn typed_masked(name: &str) -> Option<(bool, Type, usize)> {
    let (store, rest) = if let Some(r) = name.strip_prefix("load_masked_") {
        (false, r)
    } else {
        (true, name.strip_prefix("store_masked_")?)
    };
    let (elem, width) = match rest {
        "i32x4" => (Type::I32, 4),
        "i32x8" => (Type::I32, 8),
        "f32x4" => (Type::F32, 4),
        "f32x8" => (Type::F32, 8),
        "u64x4" => (Type::U64, 4),
        _ => return None,
    };
    Some((store, elem, width))
}

impl TypeChecker {
    /// v1.21.0 typed masked loads and stores: `load_masked_<t>(p, off, n)`
    /// reads lanes 0..n of `p[off..]` (zeros after), `store_masked_<t>(p,
    /// off, v, n)` writes lanes 0..n. Native on AVX2 (vpmaskmovd/q,
    /// vmaskmovps); ARM is a codegen error.
    pub(super) fn check_maskmem_intrinsic(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        let (store, elem, width) = typed_masked(name)?;
        Some(self.check_maskmem(name, store, elem, width, args, locals, span))
    }

    #[allow(clippy::too_many_arguments)]
    fn check_maskmem(
        &self,
        name: &str,
        store: bool,
        elem: Type,
        width: usize,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        let n_args = if store { 4 } else { 3 };
        if args.len() != n_args {
            return Err(CompileError::type_error(format!("{name} expects {n_args} arguments"), span.clone()));
        }
        let vec = Type::Vector { elem: Box::new(elem.clone()), width };
        let p = self.check_expr(&args[0], locals)?;
        let ok_ptr = matches!(&p, Type::Pointer { inner, mutable, .. } if inner.as_ref() == &elem && (*mutable || !store));
        if !ok_ptr {
            let want = if store { format!("*mut {elem}") } else { format!("*{elem}") };
            return Err(CompileError::type_error(format!("{name} expects a pointer {want}, got {p}"), args[0].span().clone()));
        }
        let ints = if store { [1usize, 3] } else { [1, 2] };
        for i in ints {
            let t = self.check_expr(&args[i], locals)?;
            if !(t == Type::I32 || t == Type::IntLiteral) {
                return Err(CompileError::type_error(format!("{name}: offset and count are i32, got {t}"), args[i].span().clone()));
            }
        }
        if store {
            let v = self.check_expr(&args[2], locals)?;
            if v != vec {
                return Err(CompileError::type_error(format!("{name} stores a {vec}, got {v}"), args[2].span().clone()));
            }
            return Ok(Type::Void);
        }
        Ok(vec)
    }
}
