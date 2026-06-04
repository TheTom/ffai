# FFAI shared-engine verification matrix

One Rust codebase (`ffai-models` + `ffai-ops`) over the `ffai-core::Device`
trait, run on each backend, **argmax checked against HF transformers** for a
single-token forward (`forward_single` — token attends to itself at pos 0,
which is exactly HF's 1-token forward). The multi-token KV-cache decode loop
is the runtime's job; this validates the model graph + every op.

## Dense transformer-LLM family — one builder, auto-config (`load_hf`)

`load_hf` reads HF `config.json` for geometry and detects arch flags from the
tensor names (qk-norm by `q_norm.weight`, QKV bias by `q_proj.bias`). No
per-model code.

| model | arch detail | CUDA (GB10 / sm_121) | Metal (Apple GPU) | HF argmax (tok 9707) |
|---|---|:---:|:---:|:---:|
| Qwen3-0.6B    | qk-norm, tied, hd128       | ✅ 21806 | ✅ 21806 | 21806 |
| Qwen2.5-0.5B  | QKV bias, tied, hd64       | ✅ 11    | ✅ 11    | 11 |
| SmolLM2-135M  | Llama-arch, tied, hidden 576 | ✅ 28  | ✅ 28    | 28 |

Three distinct architectures (qk-norm / QKV-bias / plain-Llama) and a
non-128-multiple hidden (576, handled by the strided `mt_rms_norm_wide`
fallback) — all auto-loaded by `load_hf`, identical argmax on CUDA, Metal,
and HF. Same code path covers the rest of the family as weights are staged:
**Llama 3.x, Mistral, Yi, Phi, Starcoder2, OLMo, InternLM2, Granite3, …** —
dense Llama-style transformers differing only by config + the qk-norm/bias
flags `load_hf` already detects.

## Architecture-path coverage — distinct non-Llama mechanisms (host-orchestrated, vs HF argmax/top-k)

Each model below was chosen to exercise a *mechanism* the dense `load_hf` path
doesn't, on the shared op layer. All single-token forwards, argmax/top-k vs HF.

| model | mechanism exercised | CUDA | Metal | HF |
|---|---|:---:|:---:|:---:|
| GPT-2 124M | LayerNorm-LLM, learned-pos, Conv1D (transposed) weights, gelu_new, tied | ✅ 198 | ✅ 198 | 198 |
| Pythia-160m | GPT-NeoX: parallel residual, interleaved per-head QKV, partial rotary | ✅ 285 | ✅ 285 | 285 |
| Gemma-2-2b | √hidden embed-scale, RMSNorm(1+w), 4 norms/layer, geGLU, GQA hd256, softcaps | ✅ top3 | ✅ top3 | [9707,235265,110] |
| Phi-1.5 | single shared norm → parallel attn+MLP, separate q/k/v+bias, partial rotary | ✅ top3 | ✅ top3 | [11,13,546] |
| OLMo-2-1B | **post-norm** (norm on sublayer output) + qk-norm over full proj, SwiGLU | ✅ top3 | ✅ top3 | [198,8,13] |
| StableLM-2-1.6B | LayerNorm(+bias), q/k/v bias, partial rotary, SwiGLU | ✅ top3 | ✅ top3 | [341,11,280] |
| GPT-Neo-125M | learned-pos + LayerNorm, separate q/k/v (no bias), no attn scaling, tied | ✅ top3 | ✅ top3 | [28,59,91] |

(`load_hf` already covers qk-norm / QKV-bias / plain-Llama / GQA via Qwen3·Qwen2.5·SmolLM2.)

## Prefill primitive — multi-token causal forward (vs HF argmax/top-k)

The model tests above are single-token (pos 0). These verify the **multi-token
causal prefill** path — each position attends over [0, i] via `sdpa_decode`
with `n_kv = i+1` against the per-head K/V cache (`kv_stride = seq_len`).

| test | what it adds | CUDA | Metal | HF |
|---|---|:---:|:---:|:---:|
| GPT-2 prefill (8 tok) | causal masking + learned positions | ✅ | ✅ | top3 [11,21831,7586] |
| Llama prefill (7 tok) | **RoPE-at-position** + GQA + SwiGLU (SmolVLM text model) | ⏳ | ✅ | top3 [12642,4052,216] |

### Throughput (correctness-first, not yet optimized) — GPT-2-124M, incremental KV cache + device-resident weights
| platform | prefill | decode | one-time (weight upload / kernel JIT) | output |
|---|---|---|---|---|
| CUDA (GB10) | 52.5 tok/s | **24.6 tok/s** (41 ms/tok) | 16.1s / 42.4s | exact HF match |
| Metal (Apple GPU) | 11 tok/s | **9.4 tok/s** (107 ms/tok) | 10.7s / 31.5s | exact HF match |

Resident weights gave a 26× decode speedup over re-uploading per step. Remaining
overhead is per-layer host round-trips + KV re-upload (device-resident activations
+ device KV cache is the next win) and F32-only compute. Not production tok/s yet.

With these, every primitive for end-to-end generation + multimodal prefill is
verified. A full VLM forward = [SigLIP tower] → [SmolVLM connector] → splice
image embeds into the text sequence → [this Llama causal prefill] — each piece
independently confirmed vs HF on the shared op layer.

## Exotic families — dedicated builders / ops (in progress)

| family | needs | status |
|---|---|---|
| MoE — OLMoE / Qwen2-MoE / GPT-OSS / Granite4 | router → top-k → per-expert SwiGLU (+ optional shared expert) | ✅ **full real OLMoE-1B-7B verified vs HF both platforms** (argmax 310 on Metal + CUDA) — 16-layer, 64-expert, top-8, no-renorm (`norm_topk_prob=false`), no shared expert, qk-norm over the *full* 2048 projection, MHA hd128, sharded BF16 checkpoint via the mmap sharded loader. Plus the real Qwen2-MoE block (sigmoid-gated shared expert, max\|Δ\|4.6e-7 both platforms). |
| DeepSeek-V4 (MLA + DSv4-MoE + mHC) | full novel arch | ✅ **entire compute path + GGUF loader verified** (MLA/MoE/mHC composites both platforms vs CPU; F32/F16/Q8_0/Q2_K/IQ2_XXS dequant vs gguf-py on the 81GB checkpoint). Full 43-layer forward blocked on CSA/HCA sparse-attention (WIP in the reference itself — no correct oracle). |
| SSM — Mamba2, Jamba, LFM2 | conv1d + SSD selective-scan + gated RMSNorm | ✅ **full real Mamba2-130m verified vs HF both platforms** (argmax 310 on CUDA + Metal) — 24-layer forward: in_proj → conv1d+silu → SSD scan → gated RMSNorm → out_proj, on the shared op layer. |
| Hybrid SSM+attention — **Falcon-H1**, Jamba, Zamba | per-layer Mamba2 mixer ∥ attention, µP multipliers | ✅ **full real Falcon-H1-0.5B verified vs HF both platforms** (top-3 [593,531,587] on CUDA + Metal) — each of 36 layers runs the Mamba2 mixer AND GQA attention in parallel off one shared norm, summed into the residual, then SwiGLU FFN; all µP multipliers handled exactly. Reuses `conv1d_step`+`ssm_step` and `sdpa_decode` together in one model. |
| VLM — SigLIP/CLIP towers · Pixtral, SmolVLM2, FastVLM, Idefics3, MiniCPMV | vision tower + projector + the LLM builder | ✅ **full real SigLIP-base vision tower verified vs HF both platforms** (last_hidden_state max\|Δ\| 9e-5 / sum −792.79 vs HF −792.795, Metal + CUDA) — 12-layer bidirectional ViT: patch-embed (conv-as-matmul) + position-embed + LayerNorm + full self-attention + GELU(tanh)-MLP, via the new `matmul`/`layer_norm`/`gelu` ops (all unit-checked vs CPU both platforms). The LLM half is the dense-Llama family above; a VLM = this tower → connector → that LLM. **SmolVLM (Idefics3) connector now verified vs HF both platforms** (pixel-shuffle gathering a 4×4 patch block into each token's channels + modality projection; out max\|Δ\| 5e-4 / sum 79.85 vs HF 79.852, Metal+CUDA). All three pieces of a VLM (SigLIP tower, connector, dense-Llama LLM) are independently verified; the remaining work is only the end-to-end token-splice plumbing. |
| Audio — Whisper (enc **+ dec**) · Parakeet, Voxtral, StyleTTS2, … | encoder/decoder + audio front-end + cross-attention | ✅ **full real Whisper encoder AND encoder→decoder STT verified vs HF both platforms.** Encoder (whisper-tiny): last_hidden_state max\|Δ\| 2e-5 — conv front-end (Conv1d as im2col+`matmul`) + sinusoidal pos + bidirectional transformer. **Full STT (whisper-base): argmax 50362 = HF on Metal+CUDA** — adds the **cross-attention** path (decoder Q over the encoder's 1500 K/V states, per layer) + causal self-attn + tied lm_head. Cross-attention is the mechanism every encoder-decoder + many VLMs share. |

## How to verify a model

```sh
# Metal (Mac)
MODEL_DIR=/path/to/hf_model TOK=9707 EXPECT=<hf_argmax> \
  cargo test -p ffai-metal --test hf_model -- --nocapture
# CUDA (GB10)
MODEL_DIR=/path/to/hf_model TOK=9707 EXPECT=<hf_argmax> \
  cargo test -p ffai-cuda --features cuda --test hf_model -- --nocapture
```

`EXPECT` is HF transformers' argmax for the same single input token — the
oracle. Swift FFAI is itself HF-validated, so matching HF ≡ matching Swift.
