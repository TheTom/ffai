// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-loader
//!
//! Weight loaders. Pure CPU byte-parsing + upload through the
//! [`Device`](ffai_core::Device) trait — no GPU API, fully shared across
//! backends. SafeTensors is implemented; GGUF / HF follow.

use ffai_core::{DType, Error, Result};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One tensor's location + metadata inside a SafeTensors blob.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// A memory-resident SafeTensors file. Holds the whole blob; `tensor()`
/// returns a zero-copy slice of a weight.
pub struct SafeTensors {
    bytes: Arc<Vec<u8>>,
    data_start: usize,
    index: BTreeMap<String, TensorInfo>,
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
    /// Open + parse a `.safetensors` file (8-byte header length, header JSON,
    /// then the tightly-packed data section).
    pub fn open(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::Msg(format!("read {path}: {e}")))?;
        if bytes.len() < 8 {
            return Err(Error::Msg("safetensors: file too small".into()));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])
            .map_err(|e| Error::Msg(format!("safetensors header JSON: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| Error::Msg("safetensors: header not an object".into()))?;

        let mut index = BTreeMap::new();
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
            index.insert(name.clone(), TensorInfo { dtype, shape, begin, end });
        }

        Ok(SafeTensors { bytes: Arc::new(bytes), data_start: header_end, index })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name)
    }

    /// Raw bytes + dtype + shape of a tensor, or an error if absent.
    pub fn tensor(&self, name: &str) -> Result<(&[u8], DType, &[usize])> {
        let info = self
            .index
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors: tensor '{name}' not found")))?;
        let s = self.data_start + info.begin;
        let e = self.data_start + info.end;
        Ok((&self.bytes[s..e], info.dtype, &info.shape))
    }
}
mod iq2xxs_tables;
pub mod gguf;
