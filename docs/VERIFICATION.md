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

## Exotic families — dedicated builders / ops (in progress)

| family | needs | status |
|---|---|---|
| MoE — OLMoE / Qwen2-MoE / GPT-OSS / Granite4 | router → top-k → per-expert SwiGLU (+ optional shared expert) | ✅ **full real OLMoE-1B-7B verified vs HF both platforms** (argmax 310 on Metal + CUDA) — 16-layer, 64-expert, top-8, no-renorm (`norm_topk_prob=false`), no shared expert, qk-norm over the *full* 2048 projection, MHA hd128, sharded BF16 checkpoint via the mmap sharded loader. Plus the real Qwen2-MoE block (sigmoid-gated shared expert, max\|Δ\|4.6e-7 both platforms). |
| DeepSeek-V4 (MLA + DSv4-MoE + mHC) | full novel arch | ✅ **entire compute path + GGUF loader verified** (MLA/MoE/mHC composites both platforms vs CPU; F32/F16/Q8_0/Q2_K/IQ2_XXS dequant vs gguf-py on the 81GB checkpoint). Full 43-layer forward blocked on CSA/HCA sparse-attention (WIP in the reference itself — no correct oracle). |
| SSM — Mamba2, Jamba, FalconH1, LFM2 | conv1d + SSD selective-scan + gated RMSNorm | ✅ **full real Mamba2-130m verified vs HF both platforms** (argmax 310 on CUDA + Metal) — 24-layer forward: in_proj → conv1d+silu → SSD scan → gated RMSNorm → out_proj, on the shared op layer. |
| VLM — SigLIP/CLIP towers · Pixtral, SmolVLM2, FastVLM, Idefics3, MiniCPMV | vision tower + projector + the LLM builder | ✅ **full real SigLIP-base vision tower verified vs HF both platforms** (last_hidden_state max\|Δ\| 9e-5 / sum −792.79 vs HF −792.795, Metal + CUDA) — 12-layer bidirectional ViT: patch-embed (conv-as-matmul) + position-embed + LayerNorm + full self-attention + GELU(tanh)-MLP, via the new `matmul`/`layer_norm`/`gelu` ops (all unit-checked vs CPU both platforms). The LLM half is the dense-Llama family above; a VLM = this tower → projector → that LLM. Projector + multimodal stitch pending. |
| Audio — Whisper · Parakeet, Voxtral, StyleTTS2, … | encoder/decoder + audio front-end | ✅ **full real Whisper-tiny audio encoder verified vs HF both platforms** (last_hidden_state max\|Δ\| 2e-5 / sum 12390.6 vs HF 12390.46, Metal + CUDA) — conv front-end (two Conv1d as im2col + `matmul`) + sinusoidal pos-embed + 4-layer bidirectional transformer (LayerNorm + self-attention + exact-erf GELU-MLP). Same shared op set as the VLM tower; conv-as-im2col-matmul is the reusable audio/vision front-end pattern. |

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
