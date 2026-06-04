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
| MoE / MLA — DeepSeek-V4, GPT-OSS, Granite4 | expert routing + `moe_gather_qmm` (kernels exist, benched) + MLA attention | builder pending |
| SSM — Mamba2, Jamba, FalconH1, LFM2 | gated-delta / SSM scan kernels (exist) | builder pending |
| VLM — Pixtral, SmolVLM2, FastVLM, Idefics3, MiniCPMV | vision tower + projector + the LLM builder | pending |
| Audio — TTS/STT (Parakeet, Voxtral, StyleTTS2, …) | encoder/decoder + audio front-end | pending |

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
