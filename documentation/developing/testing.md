# Testing

122 tests across 31 suites; 80.8% line coverage at the Phase 4
checkpoint. Every kernel, every Swift function, every model layer
gets a unit test. CI gates on coverage + correctness.

## Running tests

```bash
make test                                       # everything (~30s)
swift test --filter FFAITests                   # one test target
swift test --filter LlamaGenerateTests          # one suite
swift test --filter Llama                       # pattern across suites
```

`make test` always runs `make regenerate-kernels` first so you're
never testing against stale kernels.

## Coverage

```bash
make coverage                                    # tests + summary
```

This runs `swift test --enable-code-coverage` and then drives
`xcrun llvm-cov` to print a per-file table, excluding `.build/`,
`Tests/`, and the generated `MetalTileKernels.swift`.

The 100% target documented in [`planning/plan.md`](../../planning/plan.md)
applies to the Swift surface (`Sources/FFAI/` + `Sources/MetalTileSwift/`).
Current coverage is 80.8% — the gap is mostly defensive error paths
(`fatalError` on programmer bugs, unreachable `default:` cases) that
are excluded from the denominator with `// coverage:ignore` markers.

## Test layout

```
Tests/
  MetalTileSwiftTests/   One test file per kernel. Numerical correctness
                         vs CPU reference or fixed-input/fixed-output
                         vectors, across fp32 / fp16 / bf16.

  FFAITests/             Tensor, Module, Linear, BufferPool, KVCache,
                         Sampling, ModelDownloader, ModelLifecycle,
                         Capability, ModelConfig, SafeTensors, …

  ModelTests/            One folder per model family.
    Llama/               LlamaForwardTests + LlamaGenerateTests.
    Qwen3/               Qwen3ForwardTests + Qwen3GenerateTests.

  Fixtures/              Golden activations + token sequences captured
                         from mlx-lm. Loaded at test time; never
                         re-captured during `swift test`.
```

## Golden fixtures (the testing reference convention)

Numerical references for tests are **golden fixtures**, not live
Python invocations. This keeps `swift test` fully reproducible on a
stock Apple Silicon CI runner with **zero Python dependency**.

```
Tools/capture-fixtures.py     Python script — only used to GENERATE
                              fixtures, never run during swift test.
                              Uses mlx-lm for text-only models and
                              mlx-vlm for vision-language models.

Tests/Fixtures/<model>/       Captured activations + token sequences.
  metadata.json               mlx-lm / mlx-vlm version + capture date
                              for reproducibility.
```

`mlx-vlm` lists `mlx-lm` as a runtime dependency, so a single
`pip install mlx-vlm` covers both backends. The capture script picks
the right one per model based on whether the config declares a
vision encoder.

When a fixture needs regeneration:

1. `pip install mlx-vlm` (also installs `mlx-lm`).
2. Run `python Tools/capture-fixtures.py --model <repo> --output Tests/Fixtures/<name>/`.
3. Commit the new files alongside the model change.
4. Update `metadata.json` with the `mlx-lm` / `mlx-vlm` version + date.

## Writing a test

Layer tests live in `FFAITests/`:

```swift
import XCTest
import FFAI
import Metal

final class LinearTests: XCTestCase {
    func testForwardMatchesCPUReference() throws {
        let device = Device.shared
        let weight = Tensor.from([[1, 2], [3, 4]], device: device)
        let input = Tensor.from([1, 1], device: device)
        let layer = Linear(weight: weight)
        let out = layer.forward(input)
        XCTAssertEqual(out.toCPU(), [3, 7], accuracy: 1e-5)
    }
}
```

Model tests live in `ModelTests/<Family>/` and load the model + a
golden fixture:

```swift
final class LlamaGenerateTests: XCTestCase {
    func testDeterministicGreedyGreedy() async throws {
        let model = try await Model.load("unsloth/Llama-3.2-1B")
        let result = try await model.generate(
            prompt: "Once upon a time",
            options: GenerateOptions(maxNewTokens: 16)
        )
        let golden = try loadFixture("llama-3.2-1b-once-upon.json")
        XCTAssertEqual(result.generatedTokens, golden.tokens)
    }
}
```

## CI

`.github/workflows/ci.yml` runs on Apple Silicon, executes
`swift test`, uploads the coverage report, and fails any PR that
drops coverage below the configured threshold.

`.github/workflows/auto-label.yml` applies conventional-commit PR
labels (adapted from mlx-swift-lm).

## What we don't test

- **Property / fuzz testing.** Out of scope for v0.1; revisit later.
- **GPU mocking.** All tests run real Metal dispatches.
- **Defensive `fatalError` on programmer bugs.** Excluded from
  coverage via `// coverage:ignore`.
- **Multi-GPU / Linux / CUDA.** Different project.

## See also

- [Developing](developing.md) — the `make` workflow, kernel
  regeneration.
- [Adding a model](adding-a-model.md) — including which tests to add.
- [Performance](../performance.md) — `Tests/PerfTests/` regression
  thresholds.
