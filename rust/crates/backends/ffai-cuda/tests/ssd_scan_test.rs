#![cfg(feature = "cuda")]
// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! Mamba2 SSD chunked-MATMUL prefill scan (`ssm_prefill_scan_ssd`) on CUDA vs
//! the sequential `ssm_prefill_scan` reference, for the NemotronH cell
//! (dh=64, ds=128, H=64, G=8). Gate: max|Δy| < 1e-3 (fp16 GEMM path).
use ffai_core::{DType, Device, Tensor};
use ffai_cuda::CudaDevice;
use ffai_ops::{ssm_prefill_scan, ssm_prefill_scan_ssd};

fn tb(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn fb(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
// small, smooth pseudo-random fill in a sane range (keeps decay finite).
fn fill(n: usize, s: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i * 31 + s * 977) % 251) as f32 / 251.0 - 0.5) * 2.0 * scale).collect()
}
fn tn(d: &dyn Device, v: &[f32], sh: Vec<usize>) -> Tensor {
    Tensor::new(d.upload(&tb(v)).unwrap(), sh, DType::F32)
}

fn run_for_len(dev: &dyn Device, t: usize, l: u32) {
    let (h, dh, ds, ng) = (64usize, 64usize, 128usize, 8usize);
    // Use REALISTIC Mamba2 magnitudes to flush out precision/structural bugs the
    // benign synthetic ranges hide: bigger x/B/C, a wide A spread, and dt up to
    // ~1.5 (post-softplus) so decay spans a large dynamic range.
    let x = fill(t * h * dh, 1, 4.0);
    // a_log spread so A=-exp(a_log) ranges ~[-0.05, -30] across heads.
    let a_log: Vec<f32> = (0..h).map(|i| -3.0 + 6.0 * (i as f32 / h as f32)).collect();
    let b = fill(t * ng * ds, 3, 3.0);
    let c = fill(t * ng * ds, 4, 3.0);
    let dsk = fill(h, 5, 1.0);
    let dt: Vec<f32> = (0..t * h).map(|i| 0.05 + 1.4 * (((i * 13) % 11) as f32 / 11.0)).collect();
    let si = vec![0.0f32; h * dh * ds]; // prefill starts at zero state

    let xt = tn(dev, &x, vec![t * h * dh]);
    let at = tn(dev, &a_log, vec![h]);
    let bt = tn(dev, &b, vec![t * ng * ds]);
    let ct = tn(dev, &c, vec![t * ng * ds]);
    let dt_t = tn(dev, &dsk, vec![h]);
    let dtt = tn(dev, &dt, vec![t * h]);
    let sit = tn(dev, &si, vec![h * dh * ds]);

    // Sequential reference (on device).
    let (so_seq, y_seq) = ssm_prefill_scan(
        dev, &xt, &at, &bt, &ct, &dt_t, &dtt, &sit,
        t as u32, dh as u32, ds as u32, h as u32, ng as u32,
    ).unwrap();
    // SSD matmul path (on device).
    let (so_ssd, y_ssd) = ssm_prefill_scan_ssd(
        dev, &xt, &at, &bt, &ct, &dt_t, &dtt, &sit,
        t as u32, dh as u32, ds as u32, h as u32, ng as u32, l,
    ).unwrap();
    dev.synchronize().unwrap();

    let mut yb = vec![0u8; t * h * dh * 4];
    dev.download(y_seq.buffer.as_ref(), &mut yb).unwrap();
    let y_ref = fb(&yb);
    dev.download(y_ssd.buffer.as_ref(), &mut yb).unwrap();
    let y_got = fb(&yb);
    let mut sb = vec![0u8; h * dh * ds * 4];
    dev.download(so_seq.buffer.as_ref(), &mut sb).unwrap();
    let s_ref = fb(&sb);
    dev.download(so_ssd.buffer.as_ref(), &mut sb).unwrap();
    let s_got = fb(&sb);

    let mut ey = 0.0f32;
    let mut yref_mag = 0.0f32;
    for i in 0..y_ref.len() {
        ey = ey.max((y_got[i] - y_ref[i]).abs());
        yref_mag = yref_mag.max(y_ref[i].abs());
    }
    let mut es = 0.0f32;
    let mut sref_mag = 0.0f32;
    for i in 0..s_ref.len() {
        es = es.max((s_got[i] - s_ref[i]).abs());
        sref_mag = sref_mag.max(s_ref[i].abs());
    }
    // cosine similarity on y (robust to magnitude).
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..y_ref.len() {
        dot += y_ref[i] as f64 * y_got[i] as f64;
        na += (y_ref[i] as f64).powi(2);
        nb += (y_got[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    let rel = ey / yref_mag.max(1e-6);
    let rel_s = es / sref_mag.max(1e-6);
    eprintln!(
        "SSD vs sequential  T={t} L={l}:  rel|Δy|={rel:.3e}  cos={cos:.6}  rel|Δstate|={rel_s:.3e}  (|y|max={yref_mag:.3e})"
    );
    // fp16 GEMM (f32 accumulate) → ~1e-3 relative is the floor; gate on relative + cosine.
    assert!(rel < 3e-3, "y relative mismatch too large at T={t} L={l}: rel={rel:.3e} (abs={ey:.3e})");
    assert!(cos > 0.9999, "y cosine too low at T={t} L={l}: {cos:.6}");
    assert!(rel_s < 3e-3, "state relative mismatch too large at T={t} L={l}: rel={rel_s:.3e}");
}

#[test]
fn ssd_matmul_matches_sequential() {
    let Some(dev) = CudaDevice::create().expect("cuda") else {
        eprintln!("no CUDA — skip");
        return;
    };
    // exact multiples of L, and a non-multiple (tail-padded chunk).
    run_for_len(dev.as_ref(), 256, 128);
    run_for_len(dev.as_ref(), 512, 128);
    run_for_len(dev.as_ref(), 512, 256);
    run_for_len(dev.as_ref(), 300, 128); // 300 = 2*128 + 44 → tail chunk
    eprintln!("✅ SSD chunked-matmul scan matches sequential scan to <1e-3 (fp16).");
}
