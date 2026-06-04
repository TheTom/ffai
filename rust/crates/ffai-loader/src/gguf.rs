// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! GGUF v3 reader + dequant. Parses the header (metadata + tensor infos) and
//! the tightly-packed data section, and dequantizes tensors to f32. The
//! tractable formats (F32, F16, Q8_0) are implemented; the codebook k-quants
//! DSv4 also uses (Q2_K, IQ2_XXS) are the remaining work.

use ffai_core::{Error, Result};
use std::collections::BTreeMap;

/// GGML tensor type (subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Q2K,
    Iq2Xxs,
    Other(u32),
}
impl GgmlType {
    fn from_u32(t: u32) -> Self {
        match t {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            16 => GgmlType::Iq2Xxs,
            o => GgmlType::Other(o),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64, // within the data section
}

/// A memory-mapped GGUF file (handles the 81GB DSv4 checkpoint without
/// reading it into RAM).
pub struct Gguf {
    bytes: memmap2::Mmap,
    data_start: usize,
    tensors: BTreeMap<String, GgufTensor>,
    pub metadata_u32: BTreeMap<String, u32>,
    pub metadata_str: BTreeMap<String, String>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn gstr(&mut self) -> String {
        let n = self.u64() as usize;
        let s = String::from_utf8_lossy(&self.b[self.p..self.p + n]).into_owned();
        self.p += n;
        s
    }
    /// Consume a metadata value of `vtype`, returning a u32 if scalar-int or a
    /// String if string (for the few config fields we keep), else None.
    fn skip_value(&mut self, vtype: u32) -> (Option<u32>, Option<String>) {
        match vtype {
            0 | 1 => { self.p += 1; (Some(self.b[self.p - 1] as u32), None) }
            2 | 3 => { let v = u16::from_le_bytes(self.b[self.p..self.p + 2].try_into().unwrap()); self.p += 2; (Some(v as u32), None) }
            4 | 5 => (Some(self.u32()), None),
            6 => { self.p += 4; (None, None) } // f32
            7 => { self.p += 1; (Some(self.b[self.p - 1] as u32), None) } // bool
            8 => (None, Some(self.gstr())),
            10 | 11 => { let v = self.u64(); self.p += 0; (Some(v as u32), None) }
            12 => { self.p += 8; (None, None) } // f64
            9 => {
                // array: elem_type u32, len u64, then len elems
                let et = self.u32();
                let len = self.u64();
                for _ in 0..len {
                    self.skip_value(et);
                }
                (None, None)
            }
            _ => (None, None),
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut e = -1i32; let mut m = mant;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | (((e + 127 - 15) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

impl Gguf {
    pub fn open(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Msg(format!("open {path}: {e}")))?;
        // SAFETY: the file is read-only and outlives the mapping; we treat it
        // as an immutable byte slice.
        let bytes = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Msg(format!("mmap {path}: {e}")))?;
        let mut c = Cursor { b: &bytes, p: 0 };
        let magic = c.u32();
        if magic != 0x4655_4747 {
            return Err(Error::Msg(format!("not a GGUF file (magic {magic:#x})")));
        }
        let version = c.u32();
        if version != 3 {
            return Err(Error::Msg(format!("GGUF version {version} unsupported (need 3)")));
        }
        let n_tensors = c.u64();
        let n_kv = c.u64();

        let mut metadata_u32 = BTreeMap::new();
        let mut metadata_str = BTreeMap::new();
        let mut alignment: u64 = 32;
        for _ in 0..n_kv {
            let key = c.gstr();
            let vtype = c.u32();
            let (u, s) = c.skip_value(vtype);
            if let Some(u) = u {
                if key == "general.alignment" {
                    alignment = u as u64;
                }
                metadata_u32.insert(key.clone(), u);
            }
            if let Some(s) = s {
                metadata_str.insert(key.clone(), s);
            }
        }

        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = c.gstr();
            let n_dims = c.u32() as usize;
            let dims: Vec<u64> = (0..n_dims).map(|_| c.u64()).collect();
            let ggml_type = GgmlType::from_u32(c.u32());
            let offset = c.u64();
            tensors.insert(name.clone(), GgufTensor { name, dims, ggml_type, offset });
        }

        // Align the data section start.
        let data_start = c.p.next_multiple_of(alignment as usize);
        Ok(Gguf { bytes, data_start, tensors, metadata_u32, metadata_str })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.get(name)
    }

    fn raw(&self, t: &GgufTensor, n_bytes: usize) -> &[u8] {
        let s = self.data_start + t.offset as usize;
        &self.bytes[s..s + n_bytes]
    }

    /// Dequantize a tensor to f32. Supports F32, F16, Q8_0 so far.
    pub fn dequant_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name).ok_or_else(|| Error::Msg(format!("tensor '{name}' not found")))?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        match t.ggml_type {
            GgmlType::F32 => {
                let b = self.raw(t, n * 4);
                Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            }
            GgmlType::F16 => {
                let b = self.raw(t, n * 2);
                Ok(b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
            }
            GgmlType::Q8_0 => {
                // block of 32: f16 scale (2 bytes) + 32 int8.
                let nblocks = n / 32;
                let b = self.raw(t, nblocks * 34);
                let mut out = Vec::with_capacity(n);
                for blk in 0..nblocks {
                    let base = blk * 34;
                    let scale = f16_to_f32(u16::from_le_bytes([b[base], b[base + 1]]));
                    for i in 0..32 {
                        let q = b[base + 2 + i] as i8;
                        out.push(scale * q as f32);
                    }
                }
                Ok(out)
            }
            GgmlType::Q2K => {
                // block_q2_K (QK_K=256, 84 bytes): scales[16] + qs[64] + d(f16) + dmin(f16).
                const QK: usize = 256;
                let nblocks = n / QK;
                let b = self.raw(t, nblocks * 84);
                let mut out = Vec::with_capacity(n);
                for blk in 0..nblocks {
                    let base = blk * 84;
                    let scales = &b[base..base + 16];
                    let qs = &b[base + 16..base + 80];
                    let d = f16_to_f32(u16::from_le_bytes([b[base + 80], b[base + 81]]));
                    let dmin = f16_to_f32(u16::from_le_bytes([b[base + 82], b[base + 83]]));
                    let mut is = 0usize;
                    let mut q_off = 0usize;
                    for _n in (0..QK).step_by(128) {
                        let mut shift = 0u8;
                        for _j in 0..4 {
                            for half in 0..2 {
                                let sc = scales[is];
                                is += 1;
                                let dl = d * (sc & 0xF) as f32;
                                let ml = dmin * (sc >> 4) as f32;
                                for l in 0..16 {
                                    let q = (qs[q_off + half * 16 + l] >> shift) & 3;
                                    out.push(dl * q as f32 - ml);
                                }
                            }
                            shift += 2;
                        }
                        q_off += 32;
                    }
                }
                Ok(out)
            }
            other => Err(Error::Msg(format!(
                "dequant '{name}': {other:?} not yet supported (IQ2_XXS pending)"
            ))),
        }
    }
}
