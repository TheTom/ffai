# Using the CLI

The `ffai` executable is a SwiftPM product, not a Homebrew formula —
there's no global install step. After cloning the repo, build it with
`swift build` and invoke it through SwiftPM (`swift run ffai …`), the
built binary path, or by symlinking onto `PATH`.

## Build

```bash
git clone https://github.com/thewafflehaus/FFAI
cd FFAI

swift build -c release        # binary lands at .build/release/ffai
```

Use `-c debug` (the SwiftPM default) for faster compile + slower run;
`-c release` for the inference numbers you'd quote.

## Run

Pick one of three invocations — they're equivalent, just trade-offs
on ergonomics.

```bash
# (a) Via SwiftPM — no setup, recompiles if the source changed.
swift run -c release ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"

# (b) Direct binary path — no recompile check, fastest start-up.
.build/release/ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"

# (c) Symlink onto PATH (one-time) so plain `ffai …` works from anywhere.
ln -s "$PWD/.build/release/ffai" /usr/local/bin/ffai
ffai generate -m unsloth/Llama-3.2-1B -p "Once upon a time"
```

`generate` is the default subcommand, so the `-m / -p` flags can be
passed directly to `ffai` (`ffai -m … -p …` is equivalent to
`ffai generate -m … -p …`).

## Subcommands

| Subcommand | One-liner | More |
|---|---|---|
| `generate` (default) | Stream a single prompt's continuation to stdout. | `ffai generate --help` |
| `inspect` | Load a model and dump architecture + tokenization + top-K logits for a fixed probe prompt. The first thing to reach for when a new model produces broken output. | `ffai inspect --help` |
| `bench` | Run a benchmark method against a model, append to a per-day report. | [benchmarking.md](benchmarking.md) |

### `inspect` — model bring-up diagnostic

When a new model checkpoint isn't producing coherent text, run
`ffai inspect <repo>` before anything else. The output is structured
in five sections:

1. **Architecture** — family + dtype + every shape the loader inferred from
   `config.json` (hidden / nLayers / nHeads / nKVHeads / head_dim / vocab
   / max_position_embeddings). Tells you instantly whether the loader
   parsed the right config.
2. **Capabilities** — what the family declares it can do vs what
   `LoadOptions` enabled.
3. **Tokenizer** — per-token decode of a fixed prompt. Catches
   tokenization regressions (wrong special-token IDs, missing merges,
   model-vs-tokenizer mismatch) at a glance.
4. **KV cache** — bytes allocated, per-layer stride, eviction policy.
   For Gemma 3 / GPT-OSS, per-layer eviction shows up here.
5. **Top-K next-token logits** — runs prefill and prints the K
   most-likely continuations of the probe prompt. NaN logits get
   flagged with a debug-checklist hint; values are model-comparable
   (e.g. for `Once upon a time, in a quiet` you want to see `" village"`,
   `" little"`, `" forest"`, `" valley"`, not `"<pad>"`).

```bash
ffai inspect -m mlx-community/gemma-3-1b-it-bf16 -p "Once upon a time, in a quiet"
# → top-5: " village" +34.0, " little" +31.25, " valley" +29.88, …
```

Pair with `--debug` (per-subsystem trace dump) and `--profiling 1`
(wallclock breakdown) for full visibility into where a problem hides.

Common cross-cutting flags (`--stats`, `--debug`, `--profiling`) are
documented in [observability.md](observability.md).

## See also

- [Quick start](quickstart.md) — the 5-line library equivalent.
- [Benchmarking](benchmarking.md) — `ffai bench --method <name>`, KLD
  comparisons, per-day report shape.
- [Installation](installation.md) — adding FFAI to your own SwiftPM
  package (no CLI required).
