#![cfg(feature = "cuda")]
// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! DeepSeek-V4 MLA primitive: partial RoPE on the rope tail, on CUDA vs a
//! CPU reference. First validated DSv4-specific op on the shared layer.

use ffai_core::{DType, Device, Tensor};
use ffai_cuda::CudaDevice;
use ffai_ops::dsv4_partial_rope;

fn tb(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn fb(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn dsv4_partial_rope_on_cuda_matches_cpu() {
    let Some(dev) = CudaDevice::create().expect("metal init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };
    const NH: usize = 4;
    const HD: usize = 512;
    const NOPE: usize = 448;
    const HALF: usize = 32; // (HD-NOPE)/2 = 32, n_rot = 64
    let pos: u32 = 7;
    let theta = 10_000.0f32;

    let qk: Vec<f32> = (0..NH * HD).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect();
    let tq = Tensor::new(dev.upload(&tb(&qk)).unwrap(), vec![NH, HD], DType::F32);

    let out = dsv4_partial_rope(
        dev.as_ref(), &tq, NH as u32, HD as u32, NOPE as u32, HALF as u32, pos, theta, false,
    )
    .unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; NH * HD * 4];
    dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);

    // CPU reference (matches the kernel's own test).
    let mut want = qk.clone();
    for head in 0..NH {
        for p in 0..HALF {
            let inv_freq = (-(p as f32) * 2.0 * theta.ln() / (2.0 * HALF as f32)).exp();
            let th = pos as f32 * inv_freq;
            let (c, s) = (th.cos(), th.sin());
            let lo = head * HD + NOPE + 2 * p;
            let hi = lo + 1;
            want[lo] = qk[lo] * c - qk[hi] * s;
            want[hi] = qk[lo] * s + qk[hi] * c;
        }
    }

    let mut err = 0.0f32;
    for i in 0..NH * HD {
        err = err.max((got[i] - want[i]).abs());
    }
    eprintln!("dsv4_partial_rope on CUDA vs CPU: max|Δ|={err:.3e}");
    assert!(err <= 1e-4, "partial_rope mismatch: {err:.3e}");
    eprintln!("✅ DSv4 partial RoPE runs on CUDA through the shared op layer, matches CPU.");
}
