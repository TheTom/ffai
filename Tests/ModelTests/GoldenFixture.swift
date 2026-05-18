// Shared helper for ModelTests/*IntegrationTests.swift — loads a JSON
// golden fixture captured by Tools/capture-fixtures.py and asserts that
// FFAI's greedy decode produces the same token IDs.
//
// Why a shared helper: every model family runs the same shape of test
// (load model → greedy decode → compare to fixture). Centralising the
// load + compare keeps the family-specific tests focused on the
// per-family sanity (shapes, attention quirks, layer counts).
//
// The fixture JSON shape is defined by `Capture` in
// Tools/capture-fixtures.py. Mirrors here.

import Foundation
import Testing
@testable import FFAI

/// Token-level golden output captured from `mlx-vlm` / `mlx-lm`.
///
/// One file per `<slug>` under `Tests/Fixtures/<slug>/golden.json`. SPM
/// copies the whole `Tests/Fixtures` tree into the `ModelTests` test
/// bundle (see `Package.swift`).
struct GoldenFixture: Decodable {
    let model: String
    let prompt: String
    let maxTokens: Int
    let promptTokenIds: [Int]
    let generatedTokenIds: [Int]
    let generatedText: String
    /// Which MLX library produced this capture: `"mlx-vlm"` or `"mlx-lm"`.
    let reference: String
    let referenceVersion: String
    let capturedAtUtc: String
    let note: String
    /// Assert at least this many leading generated tokens match. Defaults
    /// to all of them; relax with a recorded rationale only.
    let minPrefixMatch: Int

    enum CodingKeys: String, CodingKey {
        case model
        case prompt
        case maxTokens = "max_tokens"
        case promptTokenIds = "prompt_token_ids"
        case generatedTokenIds = "generated_token_ids"
        case generatedText = "generated_text"
        case reference
        case referenceVersion = "reference_version"
        case capturedAtUtc = "captured_at_utc"
        case note
        case minPrefixMatch = "min_prefix_match"
    }

    /// Load `Tests/Fixtures/<slug>/golden.json` from the test bundle.
    static func load(_ slug: String) throws -> GoldenFixture {
        guard
            let url = Bundle.module.url(
                forResource: "golden",
                withExtension: "json",
                subdirectory: "Fixtures/\(slug)"
            )
        else {
            throw GoldenFixtureError.notFound(slug: slug)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(GoldenFixture.self, from: data)
    }
}

enum GoldenFixtureError: Error, CustomStringConvertible {
    case notFound(slug: String)
    var description: String {
        switch self {
        case .notFound(let slug):
            return "fixture not found: Tests/Fixtures/\(slug)/golden.json. "
                + "Regenerate via `python Tools/capture-fixtures.py --model <id>`."
        }
    }
}

/// Compare FFAI-generated token IDs against the golden via a multi-tier
/// assertion:
///
/// 1. **Hard floor** (must pass): the model emitted the right *number* of
///    tokens (no early exit), at least one prefix token matches the
///    golden (proves the forward pass produced sensible logits, not NaN
///    or a stuck-at-0 argmax), and no run of more than
///    `maxConsecutiveRepeat` identical tokens (catches empty-kernel /
///    degenerate-loop regressions).
/// 2. **Tightened match**: matched prefix ≥ `golden.minPrefixMatch`. This
///    starts low for fp16/int4 fixtures where FMA-order drift flips the
///    argmax very early, and gets raised as we tighten parity with mlx-lm
///    per fixture.
/// 3. **Drift annotation**: the matched-prefix ratio is always printed so
///    reviewers can spot a regression (50% drift today → 10% tomorrow
///    means something changed for the worse).
///
/// The split lets us land FFAI vs mlx-lm parity *gradually* without
/// pretending bit-exact reproduction is a near-term goal — the contract
/// the FFAI integration layer actually owes is coherent, non-degenerate
/// output, not byte-for-byte determinism against another fp16/Metal
/// implementation.  When a model's drift gets fixed, raise its
/// `min_prefix_match` in `Tools/capture-fixtures.py`'s `FixtureSpec` and
/// re-capture.
///
/// `actual` is FFAI's output (greedy decode, token IDs only — not the
/// decoded text, which normalises differently across tokenisers).
func expectGoldenMatch(
    _ actual: [Int],
    against golden: GoldenFixture,
    maxConsecutiveRepeat: Int = 5,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let expected = golden.generatedTokenIds
    let minPrefix = golden.minPrefixMatch
    let n = min(actual.count, expected.count)

    // Compute the matched prefix once — used in both the diagnostic and
    // the hard-floor / tightened-match checks.
    var matched = 0
    while matched < n && actual[matched] == expected[matched] {
        matched += 1
    }
    let ratio: Double = expected.isEmpty
        ? 0 : Double(matched) / Double(expected.count)
    let pct = String(format: "%.0f%%", ratio * 100)
    print("DRIFT [\(golden.model)] matched \(matched)/\(expected.count) (\(pct)) — tightened floor = \(minPrefix)")

    // ── Tier 1: hard floor ────────────────────────────────────────────

    // Wrong number of tokens → forward pass exited early or kept going
    // (e.g. ignored maxTokens). Either is a fatal regression.
    guard actual.count == expected.count else {
        let msg = "wrong generated count: got \(actual.count), expected \(expected.count). Forward pass likely exited early or ignored maxTokens."
        Issue.record(Comment(rawValue: msg), sourceLocation: sourceLocation)
        return
    }

    // Zero matched tokens → first-step argmax landed somewhere wildly off.
    // This catches NaN / zero logits / stuck-at-token-0 regressions.
    guard matched >= 1 else {
        let msg = """
            no prefix match against \(golden.model) — even the first generated token differs. Forward pass likely produced degenerate logits.
              expected first 8: \(Array(expected.prefix(8)))
              actual first 8  : \(Array(actual.prefix(8)))
              reference: \(golden.reference) v\(golden.referenceVersion)
            """
        Issue.record(Comment(rawValue: msg), sourceLocation: sourceLocation)
        return
    }

    // No run of identical tokens longer than `maxConsecutiveRepeat`.
    // Empty kernels / numerical instability often manifests as the model
    // emitting the same token forever (`token0 token0 token0 …`).
    var run = 1
    var prev = actual[0]
    for tok in actual.dropFirst() {
        run = (tok == prev) ? run + 1 : 1
        prev = tok
        if run > maxConsecutiveRepeat {
            let msg = "degenerate output from \(golden.model): \(run) consecutive occurrences of token \(tok). Likely empty kernel / stuck argmax. Actual: \(Array(actual.prefix(16)))…"
            Issue.record(Comment(rawValue: msg), sourceLocation: sourceLocation)
            return
        }
    }

    // ── Tier 2: tightened-prefix floor ────────────────────────────────

    if matched < minPrefix {
        let firstMismatch = matched
        let context = max(0, firstMismatch - 3)..<min(n, firstMismatch + 8)
        let expectedSlice = expected[context]
        let actualSlice = actual[context.lowerBound..<min(actual.count, context.upperBound)]
        let msg = """
            tightened-prefix floor not met for \(golden.model) (prompt=\(golden.prompt.debugDescription))
              matched prefix : \(matched) (\(pct))
              required floor : \(minPrefix)
              context around index \(firstMismatch) (window \(context.lowerBound)..<\(context.upperBound)):
                expected: \(Array(expectedSlice))
                actual  : \(Array(actualSlice))
              reference: \(golden.reference) v\(golden.referenceVersion)
              (regenerate the fixture if this drift is intentional; otherwise lower
              the FixtureSpec.min_prefix_match in Tools/capture-fixtures.py with a
              recorded rationale)
            """
        Issue.record(Comment(rawValue: msg), sourceLocation: sourceLocation)
    }
}
