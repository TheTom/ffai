// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! DeepSeek-V4 MLA primitive: partial RoPE on the rope tail, on Metal vs a
//! CPU reference. First validated DSv4-specific op on the shared layer.

use ffai_core::{DType, Device, Tensor};
use ffai_metal::MetalDevice;
use ffai_ops::{dsv4_partial_rope, sdpa_decode_sink};

fn tb(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn fb(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn dsv4_partial_rope_on_metal_matches_cpu() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
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
    eprintln!("dsv4_partial_rope on Metal vs CPU: max|Δ|={err:.3e}");
    assert!(err <= 1e-4, "partial_rope mismatch: {err:.3e}");
    eprintln!("✅ DSv4 partial RoPE runs on Apple GPU through the shared op layer, matches CPU.");
}

#[test]
fn dsv4_sink_sdpa_on_metal_matches_cpu() {
    let Some(dev) = MetalDevice::create().expect("metal init") else {
        eprintln!("no Metal device — skipping");
        return;
    };
    const NQ: usize = 2;
    const HD: usize = 512;
    const NKV: usize = 64;
    const HPG: usize = 2; // n_kv_heads = 1
    let scale = 1.0f32 / (HD as f32).sqrt();

    let q: Vec<f32> = (0..NQ * HD).map(|i| ((i % 19) as f32 - 9.0) * 0.03).collect();
    let kc: Vec<f32> = (0..NKV * HD).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
    let vc: Vec<f32> = (0..NKV * HD).map(|i| ((i % 13) as f32 - 6.0) * 0.025).collect();
    let sink: Vec<f32> = vec![0.5, -0.3];

    let tq = Tensor::new(dev.upload(&tb(&q)).unwrap(), vec![NQ, HD], DType::F32);
    let tk = Tensor::new(dev.upload(&tb(&kc)).unwrap(), vec![NKV, HD], DType::F32);
    let tv = Tensor::new(dev.upload(&tb(&vc)).unwrap(), vec![NKV, HD], DType::F32);
    let ts = Tensor::new(dev.upload(&tb(&sink)).unwrap(), vec![NQ], DType::F32);

    let out = sdpa_decode_sink(dev.as_ref(), &tq, &tk, &tv, &ts, NKV as u32, NKV as u32, HPG as u32, scale).unwrap();
    dev.synchronize().unwrap();
    let mut ob = vec![0u8; NQ * HD * 4];
    dev.download(out.buffer.as_ref(), &mut ob).unwrap();
    let got = fb(&ob);

    // CPU reference (single kv head; sink extends the denominator).
    let mut want = vec![0.0f32; NQ * HD];
    for h in 0..NQ {
        let scores: Vec<f32> = (0..NKV)
            .map(|t| scale * (0..HD).map(|d| q[h * HD + d] * kc[t * HD + d]).sum::<f32>())
            .collect();
        let m0 = scores.iter().cloned().fold(f32::MIN, f32::max);
        let m = m0.max(sink[h]);
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let denom: f32 = exps.iter().sum::<f32>() + (sink[h] - m).exp();
        for t in 0..NKV {
            let p = exps[t] / denom;
            for d in 0..HD {
                want[h * HD + d] += p * vc[t * HD + d];
            }
        }
    }

    let mut err = 0.0f32;
    for i in 0..NQ * HD {
        err = err.max((got[i] - want[i]).abs());
    }
    eprintln!("dsv4 sink-SDPA on Metal vs CPU: max|Δ|={err:.3e}");
    assert!(err <= 1e-4, "sink sdpa mismatch: {err:.3e}");
    eprintln!("✅ DSv4 d512 sink-SDPA runs on Apple GPU through the shared op layer, matches CPU.");
}
