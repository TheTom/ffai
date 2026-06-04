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
use metaltile_codegen::all_kernels;
use metaltile_core::ir::{BinOpKind, IndexExpr, KernelMode, Op, Param, ParamKind, ValueId};
use metaltile_core::shape::Shape;
// Force-link the crate that *registers* the model kernels, so `inventory`
// collects mt_rms_norm / mt_gemv / … into `all_kernels()`. Without a
// reference the linker may drop the otherwise-unused dependency.
use metaltile_std as _;

/// Look up a registered metaltile kernel by name and instantiate its IR for
/// `dtype`. This is the bridge from a semantic op to the *same* kernel the
/// Swift side dispatches (generated from the one `#[kernel]` definition).
fn lookup(name: &str, dtype: DType) -> Result<Kernel> {
    all_kernels()
        .find(|e| e.name() == name)
        .map(|e| e.build(&[dtype]))
        .ok_or_else(|| Error::Msg(format!("kernel '{name}' not registered (link metaltile-std)")))
}

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

// ── Heavier ops — dispatch the registered metaltile kernels ─────────────

/// Row-wise RMS norm: `out[r] = x[r] * rsqrt(mean(x[r]²) + eps) * weight`.
/// Dispatches the registered `mt_rms_norm` reduction kernel — the same one
/// the Swift side runs. The last dim is the row width `n`; the kernel owns 4
/// elements per thread, so `n` must be a multiple of 128 and ≤ 4096 (the
/// `mt_rms_norm_wide` variant lifts this — wired later).
pub fn rms_norm(dev: &dyn Device, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("rms_norm: scalar input".into()))?;
    if n % 128 != 0 || n > 4096 {
        return Err(Error::Msg(format!(
            "rms_norm: row width {n} must be a multiple of 128 and ≤ 4096 (use the wide variant)"
        )));
    }
    let rows = x.elem_count() / n;
    let mut k = lookup("mt_rms_norm", x.dtype)?;
    k.mode = KernelMode::Reduction; // block-reduction kernel (per-row)
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let eps_buf = dev.upload(&eps.to_le_bytes())?;

    let grid = Grid { grid: [rows as u32, 1, 1], block: [(n / 4) as u32, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Buffer(eps_buf),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Matrix-vector product `mat @ vec`: `mat` is `[m, k]` row-major, `vec` is
/// `[k]`, result is `[m]`. Dispatches the registered `mt_gemv` kernel (one
/// threadgroup per output row). This is the decode-time projection path; the
/// batched/prefill cooperative matmul is a separate kernel, wired later.
pub fn gemv(dev: &dyn Device, mat: &Tensor, vec: &Tensor) -> Result<Tensor> {
    if mat.shape.len() != 2 {
        return Err(Error::Msg(format!("gemv: mat must be 2-D, got {:?}", mat.shape)));
    }
    let (m, kdim) = (mat.shape[0], mat.shape[1]);
    if vec.elem_count() != kdim {
        return Err(Error::Msg(format!(
            "gemv: vec len {} != mat K {kdim}",
            vec.elem_count()
        )));
    }
    let mut k = lookup("mt_gemv", mat.dtype)?;
    k.mode = KernelMode::Reduction; // one block per output row, dot-product reduction
    let out = Tensor::empty(dev, vec![m], mat.dtype)?;

    let grid = Grid { grid: [m as u32, 1, 1], block: [256, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(mat.buffer.clone()),
            Binding::Buffer(vec.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((kdim as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Dense matmul `a @ b`. General cooperative-matmul kernel (prefill); routes
/// to [`gemv`] when `b` is a vector. Full tiled path wired later.
pub fn matmul(_dev: &dyn Device, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::matmul (cooperative tiled path pending)"))
}

/// Softmax over the last dim. Reduction kernel (pending kernel lookup).
pub fn softmax(_dev: &dyn Device, _x: &Tensor) -> Result<Tensor> {
    Err(Error::Unimplemented("ffai_ops::softmax (pending kernel lookup)"))
}
