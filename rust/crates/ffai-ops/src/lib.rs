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
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Cache of resolved kernel IR keyed by (name, dtype). Resolving walks the
/// whole test registry building setups, which is expensive — a forward pass
/// dispatches the same handful of kernels hundreds of times, so cache them.
fn kernel_cache() -> &'static Mutex<HashMap<(String, DType), Kernel>> {
    static C: OnceLock<Mutex<HashMap<(String, DType), Kernel>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Look up a registered metaltile kernel by name and instantiate its IR for
/// `dtype`, **with the correct dispatch mode already set**. We pull it from
/// the test registry (`metaltile_std::all_tests`) rather than the raw kernel
/// registry: the test/bench setup is where each kernel's `KernelMode`
/// (Elementwise / Reduction / Grid3D …) is configured, and that mode drives
/// codegen (e.g. whether `tid` vs `gid` is defined, the `_n_elems` arg). This
/// is the same path the CUDA corpus dispatches, so we inherit its correctness.
fn lookup(name: &str, dtype: DType) -> Result<Kernel> {
    let key = (name.to_string(), dtype);
    if let Some(k) = kernel_cache().lock().unwrap().get(&key) {
        return Ok(k.clone());
    }
    for entry in metaltile_std::all_tests() {
        let t = entry.test();
        if !t.dtypes().contains(&dtype) {
            continue;
        }
        let setup = t.setup(dtype);
        let k = setup.kernel();
        if k.name == name {
            kernel_cache().lock().unwrap().insert(key, k.clone());
            return Ok(k.clone());
        }
    }
    Err(Error::Msg(format!(
        "kernel '{name}' [{dtype}] not found in the metaltile test registry"
    )))
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
    let rows = x.elem_count() / n;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let eps_buf = dev.upload(&eps.to_le_bytes())?;

    // Fast path: 4 elems/thread, needs n a multiple of 128 and ≤ 4096.
    // Otherwise the strided wide variant handles any row width.
    let (kname, block) = if n % 128 == 0 && n <= 4096 {
        ("mt_rms_norm", (n / 4) as u32)
    } else {
        ("mt_rms_norm_wide", 256u32)
    };
    let k = lookup(kname, x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [block, 1, 1] };
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
    let k = lookup("mt_gemv", mat.dtype)?;
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

/// Decode-time scaled-dot-product attention for a single query token.
///
/// Layout (matching the kernel): `q` is `[n_q_heads, head_dim]`; `k`/`v` are
/// `[n_kv_heads, kv_stride, head_dim]` (kv cache, `kv_stride` = capacity,
/// `n_kv` = filled length); `kv_head = q_head / heads_per_group` (GQA). No
/// attention sink / sliding window (dense causal). Picks the head-dim
/// specialized kernel variant. Output is `[n_q_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode(
    dev: &dyn Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_kv: u32,
    kv_stride: u32,
    heads_per_group: u32,
    scale: f32,
) -> Result<Tensor> {
    let n_q_heads = q.elem_count() / head_dim;
    let out = Tensor::empty(dev, vec![n_q_heads, head_dim], q.dtype)?;

    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());

    // Per-variant trailing constexprs (everything after head_dim, n_kv,
    // kv_stride, heads_per_group). Dense path → sink/window disabled.
    let (name, block, trailing): (&str, u32, Vec<Binding>) = match head_dim {
        64 => ("ffai_sdpa_decode_d64", 1024, vec![u(0), f(0.0), f(scale)]), // has_sink, sink_logit, scale
        96 => ("ffai_sdpa_decode_d96", 1024, vec![f(scale)]),
        128 => ("ffai_sdpa_decode", 1024, vec![u(0), u(0), u(0), f(0.0), f(scale)]), // sink_end, window_start, has_sink, sink_logit, scale
        256 => ("ffai_sdpa_decode_d256", 1024, vec![u(0), f(0.0), f(scale)]),
        512 => ("ffai_sdpa_decode_d512", 512, vec![f(scale)]),
        _ => return Err(Error::Msg(format!("sdpa_decode: unsupported head_dim {head_dim}"))),
    };

    let kern = lookup(name, q.dtype)?;
    let mut bindings = vec![
        Binding::Buffer(q.buffer.clone()),
        Binding::Buffer(k.buffer.clone()),
        Binding::Buffer(v.buffer.clone()),
        Binding::Buffer(out.buffer.clone()),
        u(head_dim as u32),
        u(n_kv),
        u(kv_stride),
        u(heads_per_group),
    ];
    bindings.extend(trailing);
    let grid = Grid { grid: [n_q_heads as u32, 1, 1], block: [block, 1, 1] };
    dev.dispatch(&kern, &bindings, grid)?;
    Ok(out)
}

/// Run a registered elementwise kernel (Grid3D, one thread per element) with
/// the given ordered bindings, producing a fresh `out`-shaped output.
fn elementwise_kernel(
    dev: &dyn Device,
    name: &str,
    dtype: DType,
    out_shape: Vec<usize>,
    mut bindings: Vec<Binding>,
) -> Result<Tensor> {
    let k = lookup(name, dtype)?;
    let out = Tensor::empty(dev, out_shape, dtype)?;
    bindings.push(Binding::Buffer(out.buffer.clone()));
    let n = out.elem_count() as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    // output binding goes in its signature position — callers pass inputs,
    // we append the single output last (matches a,…,out param order).
    dev.dispatch(&k, &bindings, grid)?;
    Ok(out)
}

/// SiLU activation `out = x * sigmoid(x)`, elementwise. Dispatches `mt_silu`.
pub fn silu(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    elementwise_kernel(dev, "mt_silu", x.dtype, x.shape.clone(), vec![Binding::Buffer(x.buffer.clone())])
}

/// Fused SwiGLU `out = silu(gate) * up`, elementwise. Dispatches `mt_swiglu`.
pub fn swiglu(dev: &dyn Device, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape != up.shape {
        return Err(Error::Msg("swiglu: gate/up shape mismatch".into()));
    }
    elementwise_kernel(
        dev,
        "mt_swiglu",
        gate.dtype,
        gate.shape.clone(),
        vec![Binding::Buffer(gate.buffer.clone()), Binding::Buffer(up.buffer.clone())],
    )
}

/// Embedding gather: `out[t, :] = table[indices[t], :]`. `table` is
/// `[vocab, dim]`, `indices` is `[n_tokens]` (u32). Dispatches `ffai_gather`.
pub fn gather(dev: &dyn Device, table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if table.shape.len() != 2 {
        return Err(Error::Msg(format!("gather: table must be 2-D, got {:?}", table.shape)));
    }
    let dim = table.shape[1];
    let n_tokens = indices.elem_count();
    let k = lookup("ffai_gather", table.dtype)?;
    let out = Tensor::empty(dev, vec![n_tokens, dim], table.dtype)?;
    let n = (n_tokens * dim) as u32;
    let grid = Grid::d1(n.div_ceil(256), 256);
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(table.buffer.clone()),
            Binding::Buffer(indices.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((dim as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// DeepSeek-V4 partial RoPE: rotate only the rope tail of each head
/// (`[n_nope .. head_dim]`, adjacent-pair / GPT-J convention) in place; the
/// nope dims pass through. `qk` is `[n_heads, head_dim]`. Full-attention
/// layers: YaRN disabled (freq_scale=1, ext_factor=0). Dispatches
/// `ffai_dsv4_partial_rope` (grid `[n_heads, half_rot, 1]`). Mutates and
/// returns a view of `qk`'s buffer (matches the Swift in-place RoPE).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_partial_rope(
    dev: &dyn Device,
    qk: &Tensor,
    n_heads: u32,
    head_dim: u32,
    n_nope: u32,
    half_rot: u32,
    position: u32,
    theta_base: f32,
    inverse: bool,
) -> Result<Tensor> {
    let k = lookup("ffai_dsv4_partial_rope", qk.dtype)?;
    let u = |x: u32| Binding::Scalar(x.to_le_bytes().to_vec());
    let f = |x: f32| Binding::Scalar(x.to_le_bytes().to_vec());
    // Bind qk as both input and output → in-place (nope dims preserved).
    let bindings = vec![
        Binding::Buffer(qk.buffer.clone()),
        Binding::Buffer(qk.buffer.clone()),
        u(head_dim),
        u(n_nope),
        u(half_rot),
        u(position),
        f(theta_base),
        u(if inverse { 1 } else { 0 }),
        f(1.0), // freq_scale (YaRN off)
        f(0.0), // ext_factor
        f(0.0), // corr_low
        f(1.0), // corr_high
    ];
    let grid = Grid { grid: [n_heads, half_rot, 1], block: [1, 1, 1] };
    dev.dispatch(&k, &bindings, grid)?;
    Ok(Tensor::new(qk.buffer.clone(), qk.shape.clone(), qk.dtype))
}

/// Softmax over the last dim, row-wise. Dispatches `mt_softmax`. Row width
/// `n` must be a multiple of 1024 (the kernel's 4-elems/thread loop).
pub fn softmax(dev: &dyn Device, x: &Tensor) -> Result<Tensor> {
    let n = *x.shape.last().ok_or_else(|| Error::Msg("softmax: scalar input".into()))?;
    if n % 1024 != 0 {
        return Err(Error::Msg(format!("softmax: row width {n} must be a multiple of 1024")));
    }
    let rows = x.elem_count() / n;
    let k = lookup("mt_softmax", x.dtype)?;
    let out = Tensor::empty(dev, x.shape.clone(), x.dtype)?;
    let grid = Grid { grid: [rows as u32, 1, 1], block: [256, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(x.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((n as u32).to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}

/// Llama-style rotary position embedding applied to a `[n_heads, head_dim]`
/// Q or K tensor at sequence `position`. Dispatches `ffai_rope_llama`. Pass
/// the freq-band knobs disabled (`scale_factor=1`, `low=1`, `high=1`,
/// `orig_max=1e9`) for vanilla RoPE.
#[allow(clippy::too_many_arguments)]
pub fn rope_llama(
    dev: &dyn Device,
    qk: &Tensor,
    position: u32,
    theta_base: f32,
    scale_factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: f32,
) -> Result<Tensor> {
    let head_dim = *qk.shape.last().ok_or_else(|| Error::Msg("rope: scalar input".into()))?;
    let n_heads = qk.elem_count() / head_dim;
    let half = head_dim / 2;
    let k = lookup("ffai_rope_llama", qk.dtype)?;
    let out = Tensor::empty(dev, qk.shape.clone(), qk.dtype)?;
    let grid = Grid { grid: [n_heads as u32, half as u32, 1], block: [1, 1, 1] };
    dev.dispatch(
        &k,
        &[
            Binding::Buffer(qk.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            Binding::Scalar((head_dim as u32).to_le_bytes().to_vec()),
            Binding::Scalar((half as u32).to_le_bytes().to_vec()),
            Binding::Scalar(position.to_le_bytes().to_vec()),
            Binding::Scalar(theta_base.to_le_bytes().to_vec()),
            Binding::Scalar(scale_factor.to_le_bytes().to_vec()),
            Binding::Scalar(low_freq_factor.to_le_bytes().to_vec()),
            Binding::Scalar(high_freq_factor.to_le_bytes().to_vec()),
            Binding::Scalar(original_max_position.to_le_bytes().to_vec()),
        ],
        grid,
    )?;
    Ok(out)
}
