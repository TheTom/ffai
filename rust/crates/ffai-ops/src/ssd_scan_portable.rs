// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # Mamba2 SSD chunked-MATMUL prefill scan — PORTABLE backend.
//!
//! Same state-space-duality chunked-matmul algorithm as the CUDA
//! `ssm_prefill_scan_ssd` (in `ssd_scan.rs`), but built ENTIRELY from portable
//! registry kernels that codegen to MSL / HIP / SPIR-V — no `dispatch_raw_cuda`,
//! no cuBLAS `cublasGemmStridedBatchedEx`. Runs on Apple hardware MMA (and HIP /
//! Vulkan / RDNA4) via the portable `ffai_gemm_batched` Reduction-mode GEMM.
//!
//! The 4 batched GEMMs (steps 1/2/4/5 below) run on `ffai_gemm_batched`
//! (batch = nc·H, the SAME `out = input · weightᵀ` contraction `ffai_gemm`
//! uses, replicated over the batch axis with per-matrix strides). The small
//! segsum / Lmask / decay / transpose / recurrence / combine kernels run on
//! the portable `ssd_*` `#[kernel]` ops in `metaltile-std::ffai::ssm`.
//!
//! Everything stays f32 (no f16 GEMM) — portability + correctness first; the
//! sequential scan also accumulates in f32, so this matches it tightly.
//!
//! Gated by `NEMOTRON_SSD_PORTABLE` at the call site (A/B-able vs the
//! sequential `ssm_prefill_scan`). The CUDA `ssm_prefill_scan_ssd` is left
//! byte-for-byte unchanged.
//!
//! Fixed for NemotronH: dh=64, ds=128, H=64, G=8 (hpg=8). L ∈ {128,256}.

use ffai_core::{Binding, DType, Device, Error, Grid, Result, Tensor};
use metaltile_core::ir::KernelMode;

use crate::cached_ir;

/// Portable Mamba2 SSD **chunked-matmul prefill scan** for the NemotronH cell
/// (dh=64, ds=128, H=64, G=8). Drop-in replacement for the sequential
/// `ssm_prefill_scan`: returns `(state_out, y)` with `y` `[T·H·dh]` f32 and
/// `state_out` matching `state_in`. Inputs are f32. `t_total` need not be a
/// multiple of `chunk_len` — the tail chunk is zero-padded. `chunk_len` is
/// typically 128 or 256. Backend-portable (runs on Metal/HIP/Vulkan).
#[allow(clippy::too_many_arguments)]
pub fn ssm_prefill_scan_ssd_portable(
    dev: &dyn Device,
    x: &Tensor,
    a_log: &Tensor,
    b_mat: &Tensor,
    c_mat: &Tensor,
    d_skip: &Tensor,
    dt: &Tensor,
    state_in: &Tensor,
    t_total: u32,
    dh: u32,
    ds: u32,
    n_heads: u32,
    n_groups: u32,
    chunk_len: u32,
) -> Result<(Tensor, Tensor)> {
    if (dh, ds, n_heads, n_groups) != (64, 128, 64, 8) {
        return Err(Error::Msg(format!(
            "ssm_prefill_scan_ssd_portable: only Nemotron cell (64,128,64,8) wired, got ({dh},{ds},{n_heads},{n_groups})"
        )));
    }
    if x.dtype != DType::F32 {
        return Err(Error::Msg(
            "ssm_prefill_scan_ssd_portable: only F32 input wired (cast first)".into(),
        ));
    }
    let (t, h, dhu, dsu, g) = (
        t_total as usize,
        n_heads as usize,
        dh as usize,
        ds as usize,
        n_groups as usize,
    );
    let hpg = h / g; // heads per group
    let l = chunk_len as usize;
    let nc = t.div_ceil(l); // number of chunks (tail zero-padded)
    let bhc = nc * h; // batch count for the GEMMs

    // ── Output + final state (f32, matching the sequential scan). ─────────
    let y = Tensor::empty(dev, vec![t * h * dhu], DType::F32)?;
    let state_out = Tensor::empty(dev, state_in.shape.clone(), DType::F32)?;

    // ── Scratch buffers (all f32). ────────────────────────────────────────
    let lcs = Tensor::empty(dev, vec![bhc * l], DType::F32)?; // [nc*H, L]
    let b_g = Tensor::empty(dev, vec![bhc * l * dsu], DType::F32)?; // [nc*H, L, ds]
    let c_g = Tensor::empty(dev, vec![bhc * l * dsu], DType::F32)?; // [nc*H, L, ds]
    let xt = Tensor::empty(dev, vec![bhc * dhu * l], DType::F32)?; // [nc*H, dh, L]
    let cb = Tensor::empty(dev, vec![bhc * l * l], DType::F32)?; // [nc*H, L, L]
    let mmask = Tensor::empty(dev, vec![bhc * l * l], DType::F32)?; // [nc*H, L, L]
    let y_intra = Tensor::empty(dev, vec![bhc * l * dhu], DType::F32)?; // [nc*H, L, dh]
    let bdt = Tensor::empty(dev, vec![bhc * dsu * l], DType::F32)?; // [nc*H, ds, L]
    let s_chunk = Tensor::empty(dev, vec![bhc * dsu * dhu], DType::F32)?; // [nc*H, ds, dh]
    let sin_t = Tensor::empty(dev, vec![bhc * dhu * dsu], DType::F32)?; // [nc*H, dh, ds]
    let cs = Tensor::empty(dev, vec![bhc * l * dhu], DType::F32)?; // [nc*H, L, dh]

    let u = |v: u32| Binding::Scalar(v.to_le_bytes().to_vec());
    let (tt, hh, gg, ll, dsd, dhd, ncn, hpgn) = (
        t_total,
        n_heads,
        n_groups,
        l as u32,
        ds,
        dh,
        nc as u32,
        hpg as u32,
    );

    // Grid3D elementwise dispatch helper: one thread per output element.
    let elem = |name: &str, bufs: &[&Tensor], scalars: &[Binding], n_threads: usize| -> Result<()> {
        let kern = cached_ir(name, DType::F32, || {
            let mut k = build_ssd_ir(name);
            k.mode = KernelMode::Grid3D;
            k
        });
        let mut bindings: Vec<Binding> =
            bufs.iter().map(|t| Binding::Buffer(t.buffer.clone())).collect();
        bindings.extend_from_slice(scalars);
        // 1-D grid; cap block at 256 threads per group.
        let block = 256u32;
        let grid = Grid {
            grid: [(n_threads as u32).div_ceil(block), 1, 1],
            block: [block, 1, 1],
        };
        dev.dispatch(&kern, &bindings, grid)
    };

    // Batched GEMM helper: out[bh][r,o] = Σ_k weight[bh][o,k]·input[bh][r,k].
    // Reduction-mode, grid [(out_dim/32),(n_rows/32), batch]. All operands f32.
    let gemm_b = |weight: &Tensor,
                  input: &Tensor,
                  out: &Tensor,
                  m: usize, // n_rows
                  n: usize, // out_dim
                  k: usize, // in_dim
                  w_stride: usize,
                  x_stride: usize,
                  o_stride: usize|
     -> Result<()> {
        let kern = cached_ir("ffai_gemm_batched", DType::F32, || {
            let mut k = metaltile_std::ffai::gemm::ffai_gemm_batched::kernel_ir_for(DType::F32);
            k.mode = KernelMode::Reduction;
            k
        });
        let bindings = vec![
            Binding::Buffer(weight.buffer.clone()),
            Binding::Buffer(input.buffer.clone()),
            Binding::Buffer(out.buffer.clone()),
            u(k as u32),        // in_dim
            u(n as u32),        // out_dim
            u(m as u32),        // n_rows
            u(w_stride as u32),
            u(x_stride as u32),
            u(o_stride as u32),
        ];
        let grid = Grid {
            grid: [
                (n as u32).div_ceil(32),
                (m as u32).div_ceil(32),
                bhc as u32,
            ],
            block: [1024, 1, 1],
        };
        dev.dispatch(&kern, &bindings, grid)
    };

    // 1. Lcs cumsum: one thread per (chunk, head).
    elem(
        "ssd_lcs",
        &[dt, a_log, &lcs],
        &[u(tt), u(hh), u(ll), u(ncn)],
        bhc,
    )?;
    // 2. Gather B/C (broadcast per head).
    elem(
        "ssd_gather_bc",
        &[b_mat, c_mat, &b_g, &c_g],
        &[u(tt), u(hh), u(gg), u(hpgn), u(ll), u(dsd), u(ncn)],
        bhc * l * dsu,
    )?;
    // 3. Transpose x → xt [nc*H, dh, L].
    elem(
        "ssd_xt",
        &[x, &xt],
        &[u(tt), u(hh), u(dhd), u(ll), u(ncn)],
        bhc * dhu * l,
    )?;

    // ── G1: CB = C · Bᵀ  [L,L] = C[L,ds]·B[L,ds]ᵀ. ────────────────────────
    // out[i,j] = Σ_s c_g[i,s]·b_g[j,s] → weight=b_g, input=c_g, k=ds.
    gemm_b(&b_g, &c_g, &cb, l, l, dsu, l * dsu, l * dsu, l * l)?;

    // 4. M = CB ⊙ Lmask ⊙ dt[j], causal.
    elem(
        "ssd_mmask",
        &[&cb, &lcs, dt, &mmask],
        &[u(tt), u(hh), u(ll), u(ncn)],
        bhc * l * l,
    )?;

    // ── G2: y_intra = M · x  [L,dh]. out[i,p]=Σ_j mmask[i,j]·xt[p,j],
    //        weight=xt[dh,L], input=mmask[L,L], k=L. ─────────────────────────
    gemm_b(&xt, &mmask, &y_intra, l, dhu, l, dhu * l, l * l, l * dhu)?;

    // 5. BdT[s,j] = decay·dt·B  [ds,L].
    elem(
        "ssd_bdt",
        &[b_mat, &lcs, dt, &bdt],
        &[u(tt), u(hh), u(gg), u(hpgn), u(ll), u(dsd), u(ncn)],
        bhc * dsu * l,
    )?;

    // ── G3: S_chunk = BdT · xt-form  [ds,dh]. out[s,p]=Σ_j bdt[s,j]·xt[p,j],
    //        weight=xt[dh,L], input=bdt[ds,L], k=L. ─────────────────────────
    gemm_b(&xt, &bdt, &s_chunk, dsu, dhu, l, dhu * l, dsu * l, dsu * dhu)?;

    // 6. Serial inter-chunk recurrence → SinT [nc*H, dh, ds] + state_out.
    elem(
        "ssd_recur",
        &[&s_chunk, &lcs, state_in, &sin_t, &state_out],
        &[u(hh), u(dhd), u(dsd), u(ll), u(ncn)],
        h * dsu * dhu,
    )?;

    // ── G4: CS = C · S_in  [L,dh]. out[i,p]=Σ_s c_g[i,s]·sin_t[p,s],
    //        weight=sin_t[dh,ds], input=c_g[L,ds], k=ds. ───────────────────
    gemm_b(&sin_t, &c_g, &cs, l, dhu, dsu, dhu * dsu, l * dsu, l * dhu)?;

    // 7. Combine → y [T,H,dh] f32.
    elem(
        "ssd_combine",
        &[&y_intra, &cs, &lcs, x, d_skip, &y],
        &[u(tt), u(hh), u(dhd), u(ll), u(ncn)],
        bhc * l * dhu,
    )?;

    Ok((state_out, y))
}

/// Resolve one of the `ssd_*` portable elementwise kernels by name. They are
/// f32-fixed (`kernel_ir_for()` with no dtype arg), Grid3D-dispatched.
fn build_ssd_ir(name: &str) -> ffai_core::Kernel {
    use metaltile_std::ffai::ssm;
    match name {
        "ssd_lcs" => ssm::ssd_lcs::kernel_ir_for(),
        "ssd_gather_bc" => ssm::ssd_gather_bc::kernel_ir_for(),
        "ssd_xt" => ssm::ssd_xt::kernel_ir_for(),
        "ssd_mmask" => ssm::ssd_mmask::kernel_ir_for(),
        "ssd_bdt" => ssm::ssd_bdt::kernel_ir_for(),
        "ssd_recur" => ssm::ssd_recur::kernel_ir_for(),
        "ssd_combine" => ssm::ssd_combine::kernel_ir_for(),
        other => panic!("build_ssd_ir: unknown ssd kernel {other}"),
    }
}
