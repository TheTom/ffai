// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Prove the shared engine seam runs a kernel end-to-end on real CUDA
//! hardware: build a metaltile `Kernel` IR, then drive it entirely through
//! the backend-neutral `ffai_core::Device` trait (alloc / upload / dispatch
//! / download / synchronize) — no CUDA-specific code in sight. This is the
//! concrete proof that CUDA consumes the shared layer.
//!
//! Runs only with `--features cuda` on a CUDA host. Skips (no failure) when
//! no device is present.
#![cfg(feature = "cuda")]

use ffai_core::{Binding, DType, Grid, Tensor};
use ffai_cuda::CudaDevice;
use metaltile_core::{
    ir::{BinOpKind, IndexExpr, Kernel, Op, Param, ParamKind, ValueId},
    shape::Shape,
};

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// out[i] = a[i] + b[i] — Elementwise f32.
fn vector_add_ir() -> Kernel {
    let mut k = Kernel::new("vector_add");
    for (name, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.name_value(ValueId::new(0), "idx");
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
        Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
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

#[test]
fn vector_add_through_shared_device_trait() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    eprintln!("device: {}", dev.name());

    const N: usize = 4096;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();

    let abuf = dev.upload(&to_bytes(&a)).unwrap();
    let bbuf = dev.upload(&to_bytes(&b)).unwrap();
    let cbuf = dev.alloc(N * 4).unwrap();

    let k = vector_add_ir();
    let grid = Grid::d1((N as u32).div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[Binding::Buffer(abuf), Binding::Buffer(bbuf), Binding::Buffer(cbuf.clone())],
        grid,
    )
    .unwrap();
    dev.synchronize().unwrap();

    let mut out = vec![0u8; N * 4];
    dev.download(cbuf.as_ref(), &mut out).unwrap();
    let c = from_bytes(&out);

    let mut max_err = 0.0f32;
    for i in 0..N {
        max_err = max_err.max((c[i] - (a[i] + b[i])).abs());
    }
    assert!(max_err <= 1e-6, "vector_add via ffai Device mismatch: max|Δ|={max_err:.3e}");
    eprintln!("vector_add through ffai_core::Device on CUDA: OK (max|Δ|={max_err:.1e})");
}

/// Drive the real op layer (`ffai_ops::add` / `mul`) on CUDA — the same
/// calls model code makes, executed through the shared Device trait.
#[test]
fn ffai_ops_elementwise_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("cuda init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    const N: usize = 2048;
    let a: Vec<f32> = (0..N).map(|i| (i % 17) as f32 - 8.0).collect();
    let b: Vec<f32> = (0..N).map(|i| (i % 5) as f32 + 1.0).collect();

    let ta = Tensor::new(dev.upload(&to_bytes(&a)).unwrap(), vec![N], DType::F32);
    let tb = Tensor::new(dev.upload(&to_bytes(&b)).unwrap(), vec![N], DType::F32);

    let sum = ffai_ops::add(dev.as_ref(), &ta, &tb).unwrap();
    let prod = ffai_ops::mul(dev.as_ref(), &ta, &tb).unwrap();
    dev.synchronize().unwrap();

    let mut sbytes = vec![0u8; N * 4];
    let mut pbytes = vec![0u8; N * 4];
    dev.download(sum.buffer.as_ref(), &mut sbytes).unwrap();
    dev.download(prod.buffer.as_ref(), &mut pbytes).unwrap();
    let s = from_bytes(&sbytes);
    let p = from_bytes(&pbytes);

    let mut err = 0.0f32;
    for i in 0..N {
        err = err.max((s[i] - (a[i] + b[i])).abs());
        err = err.max((p[i] - (a[i] * b[i])).abs());
    }
    assert!(err <= 1e-6, "ffai_ops add/mul on CUDA mismatch: max|Δ|={err:.3e}");
    eprintln!("ffai_ops::add + ffai_ops::mul on CUDA: OK (max|Δ|={err:.1e})");
}
