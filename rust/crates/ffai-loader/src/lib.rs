// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-loader
//!
//! Weight loaders. Pure CPU byte-parsing + upload through the
//! [`Device`](ffai_core::Device) trait — no GPU API, fully shared across
//! backends. SafeTensors is implemented; GGUF / HF follow.

use ffai_core::{DType, Error, Result};
use std::collections::BTreeMap;

/// One tensor's location + metadata inside a SafeTensors blob.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    shard: usize,
    begin: usize,
    end: usize,
}

/// One mmap'd `.safetensors` shard: the map + where its data section starts.
struct Shard {
    map: memmap2::Mmap,
    data_start: usize,
}

/// One or more mmap'd `.safetensors` files (single or sharded). `tensor()`
/// returns a zero-copy slice routed to the right shard. mmap keeps the 14GB+
/// sharded checkpoints off the heap.
pub struct SafeTensors {
    shards: Vec<Shard>,
    index: BTreeMap<String, TensorInfo>,
}

/// IEEE half → f32.
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

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I32" => DType::I32,
        "U32" => DType::U32,
        "I8" => DType::I8,
        "U8" => DType::U8,
        other => return Err(Error::Msg(format!("safetensors: unsupported dtype {other}"))),
    })
}

impl SafeTensors {
    /// mmap + parse one `.safetensors` shard, merging its tensors into `index`
    /// tagged with `shard_idx`. Returns the constructed [`Shard`].
    fn open_shard(path: &str, shard_idx: usize, index: &mut BTreeMap<String, TensorInfo>) -> Result<Shard> {
        let file = std::fs::File::open(path).map_err(|e| Error::Msg(format!("open {path}: {e}")))?;
        // SAFETY: read-only file outlives the mapping; treated as immutable bytes.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Msg(format!("mmap {path}: {e}")))?;
        if map.len() < 8 {
            return Err(Error::Msg("safetensors: file too small".into()));
        }
        let header_len = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        let header: serde_json::Value = serde_json::from_slice(&map[8..header_end])
            .map_err(|e| Error::Msg(format!("safetensors header JSON: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| Error::Msg("safetensors: header not an object".into()))?;
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype =
                parse_dtype(v["dtype"].as_str().ok_or_else(|| Error::Msg("missing dtype".into()))?)?;
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| Error::Msg("missing shape".into()))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = v["data_offsets"]
                .as_array()
                .ok_or_else(|| Error::Msg("missing data_offsets".into()))?;
            let begin = off[0].as_u64().unwrap() as usize;
            let end = off[1].as_u64().unwrap() as usize;
            index.insert(name.clone(), TensorInfo { dtype, shape, shard: shard_idx, begin, end });
        }
        Ok(Shard { map, data_start: header_end })
    }

    /// Open + parse a single `.safetensors` file.
    pub fn open(path: &str) -> Result<Self> {
        let mut index = BTreeMap::new();
        let shard = Self::open_shard(path, 0, &mut index)?;
        Ok(SafeTensors { shards: vec![shard], index })
    }

    /// Open a model directory: sharded (`model-XXXXX-of-YYYYY.safetensors` per
    /// `model.safetensors.index.json`) or single (`model.safetensors`). All
    /// shards are mmap'd and merged into one tensor index.
    pub fn open_dir(dir: &str) -> Result<Self> {
        let idx_path = format!("{dir}/model.safetensors.index.json");
        let files: Vec<String> = if std::path::Path::new(&idx_path).exists() {
            let txt = std::fs::read_to_string(&idx_path)
                .map_err(|e| Error::Msg(format!("read {idx_path}: {e}")))?;
            let j: serde_json::Value = serde_json::from_str(&txt)
                .map_err(|e| Error::Msg(format!("index json: {e}")))?;
            let wm = j["weight_map"].as_object()
                .ok_or_else(|| Error::Msg("index: no weight_map".into()))?;
            let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for v in wm.values() {
                if let Some(f) = v.as_str() { set.insert(format!("{dir}/{f}")); }
            }
            set.into_iter().collect()
        } else {
            vec![format!("{dir}/model.safetensors")]
        };
        let mut index = BTreeMap::new();
        let mut shards = Vec::with_capacity(files.len());
        for (i, f) in files.iter().enumerate() {
            shards.push(Self::open_shard(f, i, &mut index)?);
        }
        Ok(SafeTensors { shards, index })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name)
    }

    /// Tensor decoded to `f32` (handles F32 / F16 / BF16 on disk) + its shape.
    /// The convenience every host-orchestrated model test wants — checkpoints
    /// ship in any of the three float widths.
    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (b, dt, shape) = self.tensor(name)?;
        let v = match dt {
            DType::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            DType::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            other => return Err(Error::Msg(format!("tensor_f32 '{name}': dtype {other} unsupported"))),
        };
        Ok((v, shape.to_vec()))
    }

    /// Raw bytes + dtype + shape of a tensor, or an error if absent.
    pub fn tensor(&self, name: &str) -> Result<(&[u8], DType, &[usize])> {
        let info = self
            .index
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors: tensor '{name}' not found")))?;
        let sh = &self.shards[info.shard];
        let s = sh.data_start + info.begin;
        let e = sh.data_start + info.end;
        Ok((&sh.map[s..e], info.dtype, &info.shape))
    }
}
mod iq2xxs_tables;
pub mod gguf;
