// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-ops
//!
//! The **seam** between model code and kernels — the Rust analog of
//! FFAI-Swift's `Ops/`. Each function takes Tensors, builds (or looks up)
//! the corresponding metaltile [`Kernel`](ffai_core::Kernel), and dispatches
//! it through the [`Device`](ffai_core::Device) trait. Model code calls
//! these; it never touches a GPU API or a kernel directly. Re-implementing
//! this layer once is what lets the whole model surface above it run on
//! every backend.
//!
//! The elementwise ops below are real and run on any backend that
//! implements [`Device`] (proven on CUDA). The heavier ops (matmul /
//! rms_norm / attention) are reductions and cooperative-matmul kernels that
//! map to the registered metaltile kernel set; they land via a kernel
//! lookup and are stubbed for now.

use ffai_core::{Binding, DType, Device, Error, Grid, Kernel, Result, Tensor};
use metaltile_core::ir::{BinOpKind, IndexExpr, Op, Param, ParamKind, ValueId};
use metaltile_core::shape::Shape;

/// Build an Elementwise `out[i] = a[i] <op> b[i]` kernel for `dtype`.
fn binop_kernel(name: &str, dtype: DType, op: BinOpKind) -> Kernel {
    let mut k = Kernel::new(name);
    for (pname, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: pname.into(),
            dtype,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.push_op(
        Op::Load {
            src: "a".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            mask: None,
            other: None,
        },
        ValueId::new(1),
    );
    k.body.push_op(
        Op::Load {
            src: "b".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            mask: None,
            other: None,
        },
        ValueId::new(2),
    );
    k.body.push_op(
        Op::BinOp { op, lhs: ValueId::new(1), rhs: ValueId::new(2) },
        ValueId::new(3),
    );
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(3),
        mask: None,
    });
    k
}

/// Shared implementation for elementwise binary ops over matching-shape
/// tensors. Allocates a fresh output on `dev` and dispatches through the
/// backend-neutral [`Device`] trait.
fn elementwise(
    dev: &dyn Device,
    a: &Tensor,
    b: &Tensor,
    op: BinOpKind,
    name: &str,
) -> Result<Tensor> {
    if a.shape != b.shape {
        return Err(Error::Msg(format!(
            "{name}: shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        )));
    }
    if a.dtype != b.dtype {
        return Err(Error::Msg(format!("{name}: dtype mismatch")));
    }
    let out = Tensor::empty(dev, a.shape.clone(), a.dtype)?;
    let k = binop_kernel(name, a.dtype, op);
    let n = a.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(a.buffer.clone()),
            Binding::Buffer(b.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Elementwise sum `a + b` (e.g. residual connections).
pub fn add(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    elementwise(dev, a, b, BinOpKind::Add, "ffai_add")
}

/// Elementwise product `a * b` (e.g. gating).
pub fn mul(dev: &dyn Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    elementwise(dev, a, b, BinOpKind::Mul, "ffai_mul")
}

// ── Heavier ops — land via the registered metaltile kernel set ──────────

/// `out = x * rsqrt(mean(x²) + eps) * weight`, row-wise. Reduction kernel;
/// resolves to the registered `mt_rms_norm` family (pending kernel lookup).
pub fn rms_norm(_dev: &dyn Device, _x: &Tensor, _weight: &Tensor, _eps: f32) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::rms_norm (needs registered-kernel lookup)"))
}

/// Dense matmul `a @ b`. Cooperative-matmul kernel; resolves to the
/// registered matmul family (pending kernel lookup).
pub fn matmul(_dev: &dyn Device, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::matmul (needs registered-kernel lookup)"))
}

/// Softmax over the last dim. Reduction kernel (pending kernel lookup).
pub fn softmax(_dev: &dyn Device, _x: &Tensor) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::softmax (needs registered-kernel lookup)"))
}
