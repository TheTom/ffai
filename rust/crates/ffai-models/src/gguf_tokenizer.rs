// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! Minimal GPT-2/Qwen byte-level BPE tokenizer, read straight from GGUF
//! metadata (`tokenizer.ggml.tokens` + `.merges`). Pure CPU, no deps — the
//! same vocab the model was trained with, so encode/decode round-trips the
//! exact ids llama.cpp would produce for ASCII prompts.
//!
//! Byte-level BPE: input bytes are first mapped through the GPT-2
//! "bytes_to_unicode" table into printable Unicode chars (so e.g. a space
//! becomes 'Ġ'), then merged greedily by merge rank. Tokens in the GGUF vocab
//! are stored in that same mapped-char space, so a token string is matched
//! directly against the merged symbols.

use ffai_core::{Error, Result};
use ffai_loader::gguf::Gguf;
use std::collections::HashMap;

pub struct GgufTokenizer {
    /// id → token string (in GPT-2 mapped-char space).
    tokens: Vec<String>,
    /// token string → id.
    token_to_id: HashMap<String, u32>,
    /// "a b" merge pair → rank (lower = applied first).
    merge_rank: HashMap<(String, String), usize>,
    /// byte (0..256) → mapped char, and the inverse.
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

/// GPT-2 reversible byte→unicode table: map all 256 bytes to printable chars so
/// BPE operates on text without control/whitespace ambiguity.
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

impl GgufTokenizer {
    /// Build the tokenizer from a parsed GGUF's metadata arrays.
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let tokens = g
            .metadata_arr_str
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::Msg("gguf: tokenizer.ggml.tokens missing".into()))?
            .clone();
        let merges = g
            .metadata_arr_str
            .get("tokenizer.ggml.merges")
            .ok_or_else(|| Error::Msg("gguf: tokenizer.ggml.merges missing".into()))?;

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') {
                merge_rank.insert((a.to_string(), b.to_string()), rank);
            }
        }
        let byte_to_char = bytes_to_unicode();
        let mut char_to_byte = HashMap::with_capacity(256);
        for (b, &c) in byte_to_char.iter().enumerate() {
            char_to_byte.insert(c, b as u8);
        }

        Ok(GgufTokenizer { tokens, token_to_id, merge_rank, byte_to_char, char_to_byte })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Look up a special-token id by its literal token string (e.g.
    /// "<|im_start|>", "<|endoftext|>").
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied()
    }

    /// BPE-merge one whitespace-delimited "word" (already mapped to GPT-2
    /// char space) into the fewest tokens by merge rank.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..symbols.len() - 1 {
                if let Some(&r) = self.merge_rank.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if best.map(|(br, _)| r < br).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
        }
        symbols
    }

    /// Encode UTF-8 `text` into token ids via byte-level BPE. ASCII prompts
    /// round-trip exactly to llama.cpp's ids; this does NOT apply the GPT-2
    /// pre-tokenizer regex (word/space splitting) — instead it splits on the
    /// leading-space convention, which suffices for plain prompts.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mapped: String = text.bytes().map(|b| self.byte_to_char[b as usize]).collect();
        // Split into "words" the way GPT-2 BPE groups them: a run that starts
        // at the mapped-space char ('Ġ') and continues until the next one.
        let space = self.byte_to_char[b' ' as usize];
        let mut words: Vec<String> = Vec::new();
        let mut cur = String::new();
        for c in mapped.chars() {
            if c == space && !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            cur.push(c);
        }
        if !cur.is_empty() {
            words.push(cur);
        }

        let mut ids = Vec::new();
        for w in &words {
            for sym in self.bpe(w) {
                if let Some(&id) = self.token_to_id.get(&sym) {
                    ids.push(id);
                } else {
                    // Fall back to per-char ids (every single mapped char is in
                    // the byte-level base vocab, so this always resolves).
                    for ch in sym.chars() {
                        let s = ch.to_string();
                        if let Some(&id) = self.token_to_id.get(&s) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
        ids
    }

    /// Decode token ids back to a UTF-8 string (inverting the byte map).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.tokens.get(id as usize) {
                for c in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
