use rustc_ast::expand::typetree::{FncTree, Kind, Type, TypeTree};
use rustc_span::bug;
use tracing::trace;

use crate::ty::context::TyCtxt;
use crate::ty::{self, Ty};

/// Generate TypeTree information for autodiff.
/// This function creates TypeTree metadata that describes the memory layout
/// of function parameters and return types for Enzyme autodiff.
pub fn fnc_typetrees<'tcx>(tcx: TyCtxt<'tcx>, fn_ty: Ty<'tcx>) -> FncTree {
    // Check if TypeTrees are disabled via NoTT flag
    if tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::NoTT) {
        return FncTree { args: vec![], ret: TypeTree::new() };
    }

    // Check if this is actually a function type
    if !fn_ty.is_fn() {
        return FncTree { args: vec![], ret: TypeTree::new() };
    }

    // Get the function signature
    let fn_sig = fn_ty.fn_sig(tcx);
    let sig = tcx.instantiate_bound_regions_with_erased(fn_sig);

    // Create TypeTrees for each input parameter
    let mut args = vec![];
    for ty in sig.inputs().iter() {
        let type_tree = typetree_from_ty(tcx, *ty);
        args.push(type_tree);
    }

    // Create TypeTree for return type
    let ret = typetree_from_ty(tcx, sig.output());

    let f = FncTree { args, ret };
    f
}

/// Generate a TypeTree for a specific type.
/// Mainly a convenience wrapper around the actual implementation.
pub fn typetree_from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> TypeTree {
    if !tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::Enable) {
        return TypeTree::new();
    }
    if tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::NoTT) {
        return TypeTree::new();
    }
    let mut visited = Vec::new();
    typetree_from_ty_impl_inner(tcx, ty, 0, &mut visited, false)
}

/// Maximum recursion depth for TypeTree generation to prevent stack overflow
/// from pathological deeply nested types. Combined with cycle detection.
const MAX_TYPETREE_DEPTH: usize = 6;

fn handle_indirection<'a>(
    ty: Ty<'a>,
    tcx: TyCtxt<'a>,
    depth: usize,
    visited: &mut Vec<Ty<'a>>,
) -> TypeTree {
    let Some(inner_ty) = ty.builtin_deref(true) else {
        bug!("incorrect autodiff typetree handling for type: {}", ty);
    };
    // A pointer to a slice-tailed DST is a fat pointer `{data, len}`. `RustSlice` describes both
    // LLVM arguments, while its child describes the memory reached through `data`.
    let typing_env = ty::TypingEnv::fully_monomorphized();
    if let ty::Slice(element_ty) = tcx.struct_tail_for_codegen(inner_ty, typing_env).kind() {
        // `layout.size` here is the sized prefix of `inner_ty`, not the slice element size.
        // Direct slices, transparent wrappers (`OsStr`), and ZST-prefixed DSTs have no byte
        // offset to preserve. Nonzero prefixes (e.g. `Header<[f32]>`) keep field offsets.
        // ZST elements still take this path and yield an empty child TypeTree (size 0).
        let child = if tcx
            .layout_of(typing_env.as_query_input(inner_ty))
            .is_ok_and(|layout| layout.size.bytes() == 0)
        {
            typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false)
        } else {
            typetree_from_ty_impl_inner(tcx, inner_ty, depth + 1, visited, true)
        };
        return TypeTree(vec![Type {
            offset: -1,
            size: tcx.data_layout.pointer_size().bytes_usize(),
            kind: Kind::RustSlice,
            child,
        }]);
    }

    let child = typetree_from_ty_impl_inner(tcx, inner_ty, depth + 1, visited, true);
    return TypeTree(vec![Type {
        offset: -1,
        size: tcx.data_layout.pointer_size().bytes_usize(),
        kind: Kind::Pointer,
        child,
    }]);
}

/// Internal implementation with context about whether this is for a reference target.
fn typetree_from_ty_impl_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    depth: usize,
    visited: &mut Vec<Ty<'tcx>>,
    is_reference_target: bool,
) -> TypeTree {
    if depth >= MAX_TYPETREE_DEPTH {
        trace!("typetree depth limit {} reached for type: {}", MAX_TYPETREE_DEPTH, ty);
        return TypeTree::new();
    }

    if visited.contains(&ty) {
        return TypeTree::new();
    }
    visited.push(ty);

    let tree = match ty.kind() {
        // Direct slices are handled by `handle_indirection`. This arm describes a slice tail while
        // recursing through a prefixed DST, so its caller can add the field offset.
        ty::Slice(element_ty) => {
            typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false)
        }
        ty::Ref(..) | ty::RawPtr(..) => handle_indirection(ty, tcx, depth, visited),
        ty::Adt(def, _) if def.is_box() => handle_indirection(ty, tcx, depth, visited),
        // Strata patch: a Vec's buffer pointer is type-erased (`NonNull<u8>`), so describe it
        // directly: a pointer to elements of T, plus integer length and capacity.
        ty::Adt(def, args) if tcx.is_diagnostic_item(rustc_span::sym::Vec, def.did()) => {
            vec_typetree(tcx, ty, args.type_at(0), depth, visited)
        }
        ty::Array(element_ty, len_const) => {
            // Strata patch: a length written as a named constant may still be unevaluated here;
            // fall back to the array's size over its element's size.
            let typing_env = ty::TypingEnv::fully_monomorphized();
            let len = len_const.try_to_target_usize(tcx).unwrap_or_else(|| {
                let whole = tcx.layout_of(typing_env.as_query_input(ty)).map(|l| l.size.bytes()).unwrap_or(0);
                let one = tcx.layout_of(typing_env.as_query_input(*element_ty)).map(|l| l.size.bytes()).unwrap_or(0);
                if one > 0 { whole / one } else { 0 }
            });
            if len == 0 {
                TypeTree::new()
            } else {
                let element_tree =
                    typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false);
                let mut types = Vec::new();
                // Strata patch: offset -1 means "every element" only where the array is itself the
                // pointee; inside a struct the caller turns it into the field's offset, typing just
                // element 0. Spell out each element (bounded) so struct fields are fully typed.
                let elem_size = tcx
                    .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(*element_ty))
                    .map(|l| l.size.bytes_usize())
                    .unwrap_or(0);
                if elem_size > 0 && len as usize * element_tree.0.len() <= 4096 {
                    for i in 0..len as usize {
                        for elem_type in &element_tree.0 {
                            let off = if elem_type.offset == -1 { 0 } else { elem_type.offset };
                            types.push(Type::from_ty((i * elem_size) as isize + off, elem_type));
                        }
                    }
                } else {
                    for elem_type in &element_tree.0 {
                        types.push(Type::from_ty(-1, elem_type));
                    }
                }

                TypeTree(types)
            }
        }
        ty::Tuple(tuple_types) => {
            if tuple_types.is_empty() {
                TypeTree::new()
            } else {
                let mut types = Vec::new();
                let mut current_offset = 0;

                for tuple_ty in tuple_types.iter() {
                    let element_tree =
                        typetree_from_ty_impl_inner(tcx, tuple_ty, depth + 1, visited, false);

                    let element_layout = tcx
                        .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(tuple_ty))
                        .ok()
                        .map(|layout| layout.size.bytes_usize())
                        .unwrap_or(0);

                    for elem_type in &element_tree.0 {
                        let offset = if elem_type.offset == -1 {
                            current_offset as isize
                        } else {
                            current_offset as isize + elem_type.offset
                        };
                        types.push(Type::from_ty(offset, elem_type));
                    }

                    current_offset += element_layout;
                }

                TypeTree(types)
            }
        }
        ty::Adt(adt_def, args) if adt_def.is_struct() => {
            let struct_layout =
                tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty));
            if let Ok(layout) = struct_layout {
                let mut types = Vec::new();

                for (field_idx, field_def) in adt_def.all_fields().enumerate() {
                    let field_ty = field_def.ty(tcx, *args);
                    let field_tree = typetree_from_ty_impl_inner(
                        tcx,
                        field_ty.skip_norm_wip(),
                        depth + 1,
                        visited,
                        false,
                    );

                    let field_offset = layout.fields.offset(field_idx).bytes_usize();

                    for elem_type in &field_tree.0 {
                        let offset = if elem_type.offset == -1 {
                            field_offset as isize
                        } else {
                            field_offset as isize + elem_type.offset
                        };
                        types.push(Type::from_ty(offset, elem_type));
                    }
                }

                TypeTree(types)
            } else {
                TypeTree::new()
            }
        }
        ty::Char | ty::Bool | ty::Infer(ty::IntVar(_)) | ty::Int(_) | ty::Uint(_) => {
            let kind = Kind::Integer;
            let size = ty.primitive_size(tcx).bytes_usize();
            let offset = if is_reference_target { 0 } else { -1 };
            TypeTree(vec![Type { offset, size, kind, child: TypeTree::new() }])
        }
        ty::Float(_) | ty::Infer(ty::FloatVar(_)) => {
            let (enzyme_ty, size) = match ty {
                x if x == tcx.types.f16 => (Kind::Half, 2),
                x if x == tcx.types.f32 => (Kind::Float, 4),
                x if x == tcx.types.f64 => (Kind::Double, 8),
                x if x == tcx.types.f128 => (Kind::F128, 16),
                _ => bug!("Unexpected floating point type: {:?}", ty),
            };
            let offset = if is_reference_target { 0 } else { -1 };
            TypeTree(vec![Type { offset, size, kind: enzyme_ty, child: TypeTree::new() }])
        }
        _ => TypeTree::new(),
    };

    let popped = visited.pop();
    debug_assert_eq!(popped, Some(ty));
    tree
}

/// Byte offset of a (possibly nested) field reached by following field names from `ty`.
fn field_path_offset<'tcx>(tcx: TyCtxt<'tcx>, mut ty: Ty<'tcx>, path: &[&str]) -> Option<usize> {
    let mut offset = 0;
    for name in path {
        let ty::Adt(def, args) = ty.kind() else { return None };
        let variant = def.non_enum_variant();
        let (idx, field) = variant.fields.iter().enumerate().find(|(_, f)| f.name.as_str() == *name)?;
        let layout = tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty)).ok()?;
        offset += layout.fields.offset(idx).bytes_usize();
        ty = field.ty(tcx, *args).skip_norm_wip();
    }
    Some(offset)
}

fn vec_typetree<'tcx>(
    tcx: TyCtxt<'tcx>,
    vec_ty: Ty<'tcx>,
    elem_ty: Ty<'tcx>,
    depth: usize,
    visited: &mut Vec<Ty<'tcx>>,
) -> TypeTree {
    let ptr = field_path_offset(tcx, vec_ty, &["buf", "inner", "ptr"]);
    let len = field_path_offset(tcx, vec_ty, &["len"]);
    let cap = field_path_offset(tcx, vec_ty, &["buf", "inner", "cap"]);
    let (Some(ptr), Some(len), Some(cap)) = (ptr, len, cap) else { return TypeTree::new() };
    let usize_size = tcx.data_layout.pointer_size().bytes_usize();
    // `false`: element types at offset -1 apply to every element, as for arrays and slices
    let child = typetree_from_ty_impl_inner(tcx, elem_ty, depth + 1, visited, false);
    let mut types = vec![Type { offset: ptr as isize, size: usize_size, kind: Kind::Pointer, child }];
    for off in [len, cap] {
        types.push(Type { offset: off as isize, size: usize_size, kind: Kind::Integer, child: TypeTree::new() });
    }
    TypeTree(types)
}
