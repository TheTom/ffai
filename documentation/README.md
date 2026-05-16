# Documentation

Table of contents for the FFAI documentation. The top-level
[`README`](../README.md) is the curated landing page; this index lists
every page in the tree so you can jump straight to a topic.

## Getting started

- [Installation](installation.md) — SwiftPM / Xcode setup, platform
  requirements, sibling-metaltile checkout.
- [Quick start](quickstart.md) — generate text in 5 lines.
- [Using the CLI](using-the-cli.md) — build the `ffai` binary and run
  it via `swift run`, the built path, or a `PATH` symlink.
- [Architecture](architecture.md) — the three-layer stack
  (`metaltile` Rust → `MetalTileSwift` → `FFAI`), build pipeline, and
  per-token dispatch loop.
- [Models](models.md) — supported architectures (Llama 3.x, Qwen 3),
  per-family known gaps, and adding a new family.

## Cross-cutting topics

- [`GenerationParameters` reference](generation-parameters.md) — every
  generation knob, per-family defaults table, the three call shapes
  (default, with-override, custom).
- [Streaming](streaming.md) — `generateStream(...)`,
  `GenerationChunk` shape, cancellation, why streaming is the
  primitive over which buffered `generate(...)` is built.
- [Chat templates](chat-templates.md) — `ChatMessage` +
  `ChatTemplateOptions`, `enableThinking` / `reasoningEffort` hooks,
  per-family quirks (Qwen 3 / DeepSeek-R1 / GPT-OSS / Gemma).
- [KV cache](kv-cache.md) — the raw fp16 / bf16 cache, GPU-side
  `kv_cache_update` kernel, and what's coming (affine, TurboQuant,
  SSM/GDN).
- [Quantization](quantization.md) — mlx-format coverage (3 / 4 / 5 / 6
  / 8-bit), packing layout, sub-group split dispatch.
- [Performance](performance.md) — current `tok/s` numbers per model,
  what each Phase 4 wave got us, where the remaining headroom is.
- [Observability](observability.md) — `--stats` (per-phase memory,
  TTFT, KV cache, wired ticket), `--debug` (subsystem-tagged stderr
  logs), `--profiling` (wallclock + `os_signpost`), perplexity /
  think-vs-gen split helpers.
- [Benchmarking](benchmarking.md) — `ffai bench --method <name>` +
  `--ref-model` for KLD, per-day markdown + JSON sidecar reports
  (mlx-swift-lm-compatible row schema).
- [Capabilities & lifecycle](capabilities.md) — the
  `Capability` enum, `LoadOptions`, `ModelLifecycleEvent` stream.

## Local development

- [Developing in FFAI](developing/developing.md) — repo layout, the
  `make` workflow, regenerating kernels.
- [Adding a model](developing/adding-a-model.md) — porting a new
  architecture from a reference implementation.
- [Testing](developing/testing.md) — running tests, golden fixtures,
  coverage targets.

## See also

- Top-level [`README`](../README.md) — project landing page.
- [`planning/plan.md`](../planning/plan.md) — phased build-out, what
  ships when.
- [`planning/architecture.md`](../planning/architecture.md) —
  longer-form architecture diagrams.
