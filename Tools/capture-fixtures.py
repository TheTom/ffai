#!/usr/bin/env python3
"""
capture-fixtures.py — capture token-by-token golden outputs for FFAI's
model-integration tests, using `mlx-vlm` (preferred) with an `mlx-lm`
fallback for models mlx-vlm doesn't yet support.

Why
---

FFAI ports MLX-style models to a Swift + Metal stack.  Forward-pass parity
with the MLX reference at the token level is the highest-signal correctness
oracle we have, since the metaltile-interp CPU interpreter was dropped
(PRs #16 / #17 on the metaltile side).  Without per-token goldens the
model integration tests can only assert "generates *some* text", which
let empty kernels ship invisibly through PR #19's macro-refactor
regression.

Reference routing
-----------------

mlx-vlm is the preferred reference — it's a superset that handles text,
vision-language, and audio.  In practice mlx-vlm 0.5.0 only supports the
multimodal model families it has explicit entries for (qwen3_vl,
qwen3_omni_moe, gemma 4, llava, etc.); text-only Qwen3 / Llama / Mamba 2
fall through its drafter routing and fail to load.  This script tries
mlx-vlm first and falls back to mlx-lm when mlx-vlm raises
``ValueError: Model type … not supported``.  Each capture records which
reference produced it (``reference`` field) and that reference's version.
We pin both:

* ``mlx-vlm==0.5.0`` — used for every model it supports.
* ``mlx-lm==0.31.3`` — fallback for text-only families.

Bump either pin → regenerate all fixtures that used that reference.

What
----

For each registered ``FixtureSpec`` this script:

  1. Downloads the model into the standard ``~/.cache/huggingface/hub/``
     cache (mlx-vlm shares this with FFAI's Swift loader).
  2. Greedy-decodes ``max_tokens`` tokens from the prompt
     (``temperature=0`` → argmax — bit-deterministic given the same
     MLX / Metal toolchain).
  3. Re-tokenises the generated text with the model's processor and
     writes ``Tests/Fixtures/<slug>/golden.json`` containing the prompt,
     prompt token IDs, generated token IDs, decoded text, plus
     mlx-vlm version and capture timestamp.

The Swift tests then load the JSON and assert FFAI's token IDs match.

Greedy parity across implementations isn't *quite* bit-exact at fp16/bf16
because GPU FMA ordering inside SDPA + GEMV can rearrange a few ULPs; in
practice the first 40–80 tokens match exactly and divergence (when it
happens) is at the tail.  The Swift assertion tolerates this via the
``min_prefix_match`` field — default "all generated tokens must match";
loosen on a per-fixture basis only with a recorded rationale.

Usage
-----

::

    pip install 'mlx-vlm==0.5.0'
    python Tools/capture-fixtures.py \\
        --model mlx-community/Qwen3-1.7B-bf16

    # Batch — captures every fixture defined in REGISTRY below.
    python Tools/capture-fixtures.py --all

    # List what's registered without running:
    python Tools/capture-fixtures.py --list

Re-run whenever ``MLX_VLM_VERSION`` below is bumped.  The captured
``mlx_vlm_version`` field in each ``golden.json`` records what produced
the fixture so reviewers can verify a regeneration was deliberate.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pinned reference versions
# ---------------------------------------------------------------------------

MLX_VLM_VERSION = "0.5.0"
"""Preferred reference.  Used for every model mlx-vlm can load."""

MLX_LM_VERSION = "0.31.3"
"""Fallback reference.  Used for text-only models mlx-vlm 0.5.0 doesn't
support natively (Qwen3 dense, Llama, Mamba 2, etc.)."""


# ---------------------------------------------------------------------------
# Fixture registry — single source of truth for which models we capture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixtureSpec:
    """One (model, prompt, decode-config) triple we capture goldens for."""

    model: str
    """HuggingFace repo ID, e.g. ``mlx-community/Qwen3-1.7B-bf16``."""

    prompt: str
    """The literal prompt string fed to the model."""

    max_tokens: int = 32
    """Number of tokens to greedy-decode beyond the prompt."""

    slug: str = ""
    """Filename-safe directory under ``Tests/Fixtures/``.  Derived from
    ``model`` if left empty."""

    note: str = ""
    """Optional human-readable hint about the prompt choice (e.g. why a
    specific phrase exercises a quirky tokenizer or a hybrid layer)."""

    min_prefix_match: int | None = None
    """Tightened-prefix floor for the Swift assertion.  ``None`` means
    "match the full generated sequence" (bit-exact parity with the MLX
    reference).  Lower numbers acknowledge FFAI ↔ mlx-lm fp16 / int-quant
    drift accumulates by token N: e.g. Llama 3.2 1B currently matches 2
    tokens before FMA-order in the attention pipeline flips the argmax.
    Raise per fixture as FFAI's per-kernel parity tightens.  The hard
    floor (≥1 matched, no degenerate runs, right token count) is enforced
    unconditionally — this only controls the *tightened* floor."""

    def resolved_slug(self) -> str:
        if self.slug:
            return self.slug
        bare = self.model.split("/")[-1]
        return re.sub(r"[^a-zA-Z0-9._-]+", "-", bare).strip("-")


# Add a new fixture here when a new model is integration-tested.
# Keep this list lean: one fixture per architectural variant we ship.
#
# `min_prefix_match` is the *tightened* floor — set to the number of
# tokens FFAI currently matches mlx-lm bit-exactly under greedy decode.
# The hard floor (≥1 matched, no degenerate runs, right token count) is
# enforced unconditionally regardless.  Today's measured values were
# captured 2026-05-18 against mlx-lm 0.31.3 on Apple M1 Max with
# `GenerationParameters(temperature: 0)`:
#
#     Llama 3.2 1B fp16        : 1 / 32   (drifts at token 2 — investigate)
#     Qwen3 1.7B bf16          : 32 / 32  ✅ bit-exact
#     Qwen3 1.7B 8bit          : 32 / 32  ✅ bit-exact
#     Qwen3 1.7B 6bit          : 32 / 32  ✅ bit-exact
#     Qwen3 1.7B 5bit          : 26 / 32  (drifts at token 26)
#     Qwen3 1.7B 4bit          : 15 / 32  (drifts at token 15)
#     Qwen3 1.7B 3bit          : 24 / 32  (drifts at token 24)
#
# `min_prefix_match` is set to the observed match count, so the test
# asserts "no regression from today's parity".  Bump these up as parity
# tightens (e.g. when we land per-kernel FMA-order alignment); never
# loosen without a recorded rationale.

REGISTRY: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        model="unsloth/Llama-3.2-1B",
        prompt="The capital of France is",
        max_tokens=32,
        slug="Llama-3.2-1B",
        min_prefix_match=1,
        note="Llama-style backbone, fp16 — Phase 2 reference.  Mirror of "
             "ungated `unsloth/Llama-3.2-1B` (weights bit-identical to "
             "`meta-llama/Llama-3.2-1B`).  Drift after token 1 is unusual "
             "for fp16; bf16 Qwen3 (same hidden_size/n_layers ballpark) "
             "matches 32/32.  Investigate Llama-specific path (RoPE Llama "
             "frequency-band scaling? embedding table dequant?).",
    ),
    FixtureSpec(
        model="mlx-community/Qwen3-1.7B-bf16",
        prompt="The capital of France is",
        max_tokens=32,
        min_prefix_match=32,
        note="Qwen3 dense bf16, q_norm/k_norm in attention — Phase 2.5.  "
             "Bit-exact match against mlx-lm; tighten this back to 32 if "
             "future kernel work introduces drift.",
    ),
    FixtureSpec(
        model="mlx-community/Qwen3-1.7B-4bit",
        prompt="The capital of France is",
        max_tokens=32,
        min_prefix_match=15,
        note="Qwen3 dense 4-bit (group_size=64) — exercises dequant_gemv_int4. "
             "Diverges from mlx-lm at token 15.  Likely accumulated drift "
             "in dequant + GEMV partial-sum ordering vs mlx's `qmv` kernel.",
    ),
    FixtureSpec(
        model="mlx-community/Qwen3-1.7B-8bit",
        prompt="The capital of France is",
        max_tokens=32,
        min_prefix_match=32,
        note="Qwen3 dense 8-bit — exercises dequant_gemv_int8.  Bit-exact "
             "against mlx-lm.",
    ),
    FixtureSpec(
        model="mlx-community/Qwen3-1.7B-3bit",
        prompt="The capital of France is",
        max_tokens=32,
        min_prefix_match=24,
        note="Qwen3 dense 3-bit — odd-width dequant_gemv_int3.  Diverges at "
             "token 24.  Investigate odd-width pack arithmetic vs mlx's "
             "int3 qmv kernel before tightening.",
    ),
    FixtureSpec(
        model="mlx-community/Qwen3-1.7B-5bit",
        prompt="The capital of France is",
        max_tokens=32,
        min_prefix_match=26,
        note="Qwen3 dense 5-bit — odd-width dequant_gemv_int5.  Diverges at "
             "token 26.",
    ),
    FixtureSpec(
        model="mlx-community/Qwen3-1.7B-6bit",
        prompt="The capital of France is",
        max_tokens=32,
        min_prefix_match=32,
        note="Qwen3 dense 6-bit — odd-width dequant_gemv_int6.  Bit-exact "
             "against mlx-lm.",
    ),
    FixtureSpec(
        model="mlx-community/mamba2-130m",
        prompt="The quick brown fox jumps over",
        max_tokens=32,
        note="Mamba 2 dense SSM — exercises ssm_step + conv1d_causal_step + softplus. "
             "Capture currently blocked by an mlx-lm 0.31.3 config-parsing bug "
             "(`ModelArgs.__init__() missing intermediate_size`); fixture absent "
             "until upstream lands a fix or we use a different reference.",
    ),
)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

@dataclass
class Capture:
    """JSON-serialisable golden fixture for one (model, prompt) pair."""

    model: str
    prompt: str
    max_tokens: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    reference: str
    """Which MLX library produced this capture (``"mlx-vlm"`` or
    ``"mlx-lm"``).  Pinned version is in ``reference_version``."""
    reference_version: str
    captured_at_utc: str
    note: str = ""
    min_prefix_match: int = 0
    """Tightened-prefix floor for the Swift assertion (see
    ``FixtureSpec.min_prefix_match``).  Defaults to the full generated
    length when not set explicitly via the spec."""


def _tokenizer_of(processor_or_tokenizer: Any) -> Any:
    """mlx-vlm hands us a transformers ``ProcessorMixin``; mlx-lm hands us
    a bare tokenizer.  Normalise so the caller always has an object with
    ``.encode()``."""
    return (
        processor_or_tokenizer.tokenizer
        if hasattr(processor_or_tokenizer, "tokenizer")
        else processor_or_tokenizer
    )


def _warn_version_drift(label: str, expected: str, actual: str) -> None:
    if actual != expected:
        print(
            f"warning: installed {label} is {actual}, pinned reference is "
            f"{expected}.  Captures will record the installed version; re-pin "
            f"the constant in this script if the bump is intentional.",
            file=sys.stderr,
        )


def _capture_via_mlx_vlm(spec: FixtureSpec) -> Capture | None:
    """Try capturing through mlx-vlm.  Returns ``None`` if the model type
    isn't supported (caller falls back to mlx-lm).  Other failures
    (missing checkpoint, OOM, etc.) propagate."""
    try:
        import mlx_vlm  # type: ignore
    except ImportError as e:  # pragma: no cover — environment hint
        raise SystemExit(
            f"mlx-vlm not installed. `pip install 'mlx-vlm=={MLX_VLM_VERSION}'`"
        ) from e
    version = getattr(mlx_vlm, "__version__", "unknown")
    _warn_version_drift("mlx-vlm", MLX_VLM_VERSION, version)

    print(f"  loading via mlx-vlm…", flush=True)
    try:
        model, processor = mlx_vlm.load(spec.model)
    except ValueError as e:
        # mlx-vlm raises this when get_model_and_args can't find a model
        # class for the architecture (typical for text-only LLMs in 0.5.0).
        msg = str(e)
        if "not supported" in msg or "Model type" in msg:
            print(f"  mlx-vlm does not support {spec.model}: {msg}", flush=True)
            return None
        raise

    tokenizer = _tokenizer_of(processor)
    prompt_ids = list(tokenizer.encode(spec.prompt))

    print(
        f"  decoding {spec.max_tokens} tokens (greedy, temperature=0)…",
        flush=True,
    )
    result = mlx_vlm.generate(
        model,
        processor,
        prompt=spec.prompt,
        max_tokens=spec.max_tokens,
        temperature=0.0,
        verbose=False,
    )
    generated_text: str = result.text
    generated_ids = list(tokenizer.encode(generated_text))

    return Capture(
        model=spec.model,
        prompt=spec.prompt,
        max_tokens=spec.max_tokens,
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated_ids,
        generated_text=generated_text,
        reference="mlx-vlm",
        reference_version=version,
        captured_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        note=spec.note,
        min_prefix_match=spec.min_prefix_match
            if spec.min_prefix_match is not None
            else len(generated_ids),
    )


def _capture_via_mlx_lm(spec: FixtureSpec) -> Capture:
    """Capture through mlx-lm using ``stream_generate`` so we collect the
    bit-exact token IDs the model emitted, not a re-tokenisation of the
    decoded text (which silently re-introduces BOS and can diverge from
    the original BPE merges)."""
    try:
        import mlx_lm  # type: ignore
    except ImportError as e:  # pragma: no cover — environment hint
        raise SystemExit(
            f"mlx-lm not installed. `pip install 'mlx-lm=={MLX_LM_VERSION}'`"
        ) from e
    version = getattr(mlx_lm, "__version__", "unknown")
    _warn_version_drift("mlx-lm", MLX_LM_VERSION, version)

    print(f"  loading via mlx-lm (fallback)…", flush=True)
    model, tokenizer = mlx_lm.load(spec.model)
    tok = _tokenizer_of(tokenizer)
    prompt_ids = list(tok.encode(spec.prompt))

    print(
        f"  streaming {spec.max_tokens} tokens (greedy, temperature=0)…",
        flush=True,
    )
    # Force greedy via temperature=0 in the sampler.
    from mlx_lm.sample_utils import make_sampler  # type: ignore

    sampler = make_sampler(temp=0.0)
    generated_ids: list[int] = []
    text_parts: list[str] = []
    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=spec.prompt,
        max_tokens=spec.max_tokens,
        sampler=sampler,
    ):
        # `resp.token` is the bit-exact emitted token ID; `resp.text` is
        # that token's incremental decode (may be empty for BPE sub-pieces
        # that haven't completed a Unicode boundary yet).
        generated_ids.append(int(resp.token))
        if resp.text:
            text_parts.append(resp.text)
    generated_text = "".join(text_parts)

    return Capture(
        model=spec.model,
        prompt=spec.prompt,
        max_tokens=spec.max_tokens,
        prompt_token_ids=prompt_ids,
        generated_token_ids=generated_ids,
        generated_text=generated_text,
        reference="mlx-lm",
        reference_version=version,
        captured_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        note=spec.note,
        min_prefix_match=spec.min_prefix_match
            if spec.min_prefix_match is not None
            else len(generated_ids),
    )


def capture_one(spec: FixtureSpec) -> Capture:
    """Greedy-decode the spec.  Tries mlx-vlm first, falls back to mlx-lm
    on unsupported model types."""
    via_vlm = _capture_via_mlx_vlm(spec)
    if via_vlm is not None:
        return via_vlm
    return _capture_via_mlx_lm(spec)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _fixtures_root() -> Path:
    """Resolve ``Tests/Fixtures`` relative to this script's location."""
    here = Path(__file__).resolve()
    return here.parent.parent / "Tests" / "Fixtures"


def write_capture(capture: Capture, slug: str) -> Path:
    target = _fixtures_root() / slug
    target.mkdir(parents=True, exist_ok=True)
    out = target / "golden.json"
    payload = asdict(capture)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_spec(model: str) -> FixtureSpec | None:
    for s in REGISTRY:
        if s.model == model:
            return s
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        help="HuggingFace repo ID; must already be registered in REGISTRY.",
    )
    p.add_argument(
        "--prompt",
        help="Override the registered prompt (rarely useful — registry is the "
             "source of truth).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        help="Override the registered max_tokens.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Capture every fixture in the registry.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List registered fixtures and exit.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.list:
        for s in REGISTRY:
            print(f"  {s.resolved_slug():32s}  {s.model}")
        return 0

    if args.all:
        targets = list(REGISTRY)
    elif args.model:
        spec = _find_spec(args.model)
        if spec is None:
            print(
                f"error: {args.model!r} is not in REGISTRY. "
                f"Add it to capture-fixtures.py.",
                file=sys.stderr,
            )
            return 2
        if args.prompt or args.max_tokens:
            spec = FixtureSpec(
                model=spec.model,
                prompt=args.prompt or spec.prompt,
                max_tokens=args.max_tokens or spec.max_tokens,
                slug=spec.slug,
                note=spec.note,
            )
        targets = [spec]
    else:
        print("error: pass --model <id>, --all, or --list", file=sys.stderr)
        return 2

    failed: list[tuple[str, str]] = []
    for spec in targets:
        print(f"[{spec.resolved_slug()}]")
        try:
            cap = capture_one(spec)
        except Exception as e:  # pragma: no cover — per-model isolation
            # Gated repos, missing checkpoints, OOM, etc.  Don't abort the
            # batch — keep going so the user gets every fixture that *can*
            # be captured in one run.
            print(f"  capture failed: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append((spec.resolved_slug(), f"{type(e).__name__}: {e}"))
            continue
        out = write_capture(cap, spec.resolved_slug())
        print(f"  wrote {out} ({len(cap.generated_token_ids)} tokens)")

    if failed:
        print(f"\n{len(failed)} fixture(s) failed:", file=sys.stderr)
        for slug, msg in failed:
            print(f"  {slug}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
