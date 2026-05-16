# `GenerationParameters`

Every `Model.generate(...)` call is configured by a
`GenerationParameters` value. Each family declares its own defaults
via the family Variant protocol; the user either uses them as-is,
mutates fields, or constructs their own.

## The fields

| Field | Default | Honored today | Notes |
|---|---|---|---|
| `maxTokens: Int` | `256` | ✅ | Hard cap on generated tokens. |
| `stopOnEOS: Bool` | `true` | ✅ | Stop at the model's `eosTokenId`. |
| `extraStopTokens: Set<Int>` | `[]` | ✅ | Additional stop ids beyond EOS. |
| `prefillStepSize: Int` | `1024` | 🚧 Phase 5 | Honored once chunked prefill ships; today's per-token prefill ignores it. |
| `temperature: Float` | `0.6` | 🚧 Phase 5 | `0` → greedy. Wired through but the decode path is greedy until GPU sampling kernels land. |
| `topP: Float` | `1.0` | 🚧 Phase 5 | Nucleus cutoff. `1.0` = disabled. |
| `topK: Int` | `0` | 🚧 Phase 5 | `0` = disabled. |
| `minP: Float` | `0.0` | 🚧 Phase 5 | Qwen-style min-P cutoff. |
| `repetitionPenalty: Float` | `1.0` | 🚧 Phase 5 | `1.0` = disabled. |
| `presencePenalty: Float` | `0.0` | 🚧 Phase 5 | Additive. `0` = disabled. |
| `seed: UInt64?` | `nil` | 🚧 Phase 5 | Reproducible sampling. |

The Phase-5 fields are part of the API surface today so per-family
defaults don't churn when sampling lands. Until then they're
no-ops on the greedy fast path.

## Family defaults

Each family's Variant protocol declares a static
`defaultGenerationParameters: GenerationParameters` that captures the
values that family ships with. The `Model` instance carries the
resolved value as `model.defaultGenerationParameters`.

```swift
let model = try await Model.load("mlx-community/Qwen3-4B-4bit")

print(model.defaultGenerationParameters.topP)          // 0.95   (Qwen 3)
print(model.defaultGenerationParameters.topK)          // 20     (Qwen 3)
print(model.defaultGenerationParameters.prefillStepSize) // 1024
```

Current values:

| Family | `temperature` | `topP` | `topK` | `minP` | `repPenalty` | `prefillStepSize` | `maxTokens` |
|---|---|---|---|---|---|---|---|
| `LlamaDense` | 0.6 | 1.0 | 0 | 0.0 | 1.0 | 1024 | 256 |
| `Qwen3Dense` | 0.6 | 0.95 | 20 | 0.0 | 1.0 | 1024 | 256 |

These match mlx-swift-lm's per-family `GenerationParameters` baseline
and `defaultPrefillStepSize` for the same architectures. As new
families land (Qwen 3.5 hybrid, Qwen 3.5 MoE, Mistral, Phi, Gemma,
etc.) they declare their own defaults — see
[developing/adding-a-model.md § Step 4: family defaults](developing/adding-a-model.md#step-4--declare-family-defaults).

## Three ways to call `generate`

### 1. Family default

```swift
let result = try await model.generate(prompt: "Once upon a time")
```

`parameters` defaults to `nil`, which falls back to
`model.defaultGenerationParameters`.

### 2. Override one field

The `with(_:)` copy-mutator keeps the family-tuned baseline and
edits a single knob:

```swift
let result = try await model.generate(
    prompt: "Once upon a time",
    parameters: model.defaultGenerationParameters.with { $0.maxTokens = 64 }
)
```

This is the recommended call shape — you don't lose the family-tuned
sampling values just because you wanted a shorter generation.

### 3. Custom from scratch

```swift
let params = GenerationParameters(
    maxTokens: 1024,
    temperature: 0.0,        // greedy
    topP: 1.0,
    repetitionPenalty: 1.05
)
let result = try await model.generate(prompt: "...", parameters: params)
```

Any field you don't pass picks the `GenerationParameters.init` default,
not the family default. Use `with(_:)` (#2) when you want the
family-tuned baseline.

## CLI behaviour

`ffai --max-tokens N` overrides only the `maxTokens` field — every
other knob still picks up the family default. Omit `--max-tokens`
entirely to use the family value:

```bash
ffai --model mlx-community/Qwen3-4B-4bit --prompt "Hello"        # uses Qwen 3 defaults (256 tokens)
ffai --model mlx-community/Qwen3-4B-4bit --prompt "Hello" --max-tokens 64
```

More CLI knobs (`--temperature`, `--top-p`, etc.) land alongside the
GPU sampling kernels.

## See also

- [Quick start](quickstart.md) — basic `Model.load` + `generate`
  flow.
- [Models](models.md) — supported families and which defaults each
  carries.
- [developing/adding-a-model.md](developing/adding-a-model.md) — how
  to declare defaults when porting a new family.
- [`planning/roadmap.md`](../planning/roadmap.md) — Phase 5 sampling
  + chunked prefill commitment.
