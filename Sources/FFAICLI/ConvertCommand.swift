// Copyright 2026 Eric Kryski (@ekryski)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// `ffai convert` — quantize a bf16/fp16 HuggingFace checkpoint to
// MLX affine format using FFAI's own GPU kernels.
//
// Bit-widths are per-tensor-class: `--bits` controls the attention +
// MLP linears (the bulk of the model), and `--embedding-bits` /
// `--lm-head-bits` / `--vision-bits` override the bit-width for those
// specific tensors. Each `--*-bits` flag is optional — leave it off to
// keep that tensor full-precision (mlx-lm convention).
//
// Examples:
//   ffai convert HuggingFaceTB/SmolLM2-360M-Instruct --bits 4
//
//   # Quantize text + embeddings (both at 4-bit), keep lm_head untied
//   # at 8-bit and vision at full precision (default):
//   ffai convert <repo> --bits 4 --embedding-bits 4 --lm-head-bits 8
//
//   # Mixed: text + embed at 2-bit, vision tower at 4-bit (requires
//   # a VL tower that consumes QuantizedLinear — none ship today):
//   ffai convert <vlm> --bits 2 --embedding-bits 2 --vision-bits 4
//
//   ffai convert /local/path/to/model --bits 8 --output /tmp/out
//   ffai convert mlx-community/Llama-3.2-1B-4bit --upload-repo ekryski/my-4bit

import ArgumentParser
import FFAI
import Foundation

struct ConvertCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract: "Quantize a bf16/fp16 HuggingFace checkpoint to MLX 4-bit affine."
    )

    @Argument(
        help: "HF repo id (e.g. HuggingFaceTB/SmolLM2-360M-Instruct) or local directory path.")
    var source: String

    @Option(
        name: .shortAndLong,
        help: "Bits per weight for the main linear projections (q/k/v/o, gate/up/down): 2, 4, or 8.")
    var bits: Int = 4

    @Option(
        name: .long, help: "Output directory. Defaults to ~/.cache/ffai/converts/<repo>-<bits>bit.")
    var output: String?

    @Option(
        name: .long,
        help: "Upload to HF repo (e.g. ekryski/foo-4bit). Requires `hf` CLI authenticated.")
    var uploadRepo: String?

    @Option(
        name: .long,
        help: ArgumentHelp(
            "Bits for embed_tokens (independent of --bits). Omit to keep full-precision."))
    var embeddingBits: Int?

    @Option(
        name: .long,
        help: ArgumentHelp(
            "Bits for lm_head when untied (independent of --bits). Omit to keep full-precision."))
    var lmHeadBits: Int?

    @Option(
        name: .long,
        help: ArgumentHelp(
            "Bits for vision-tower weights (independent of --bits). Omit to keep full-precision —"
                + " FFAI VL towers run plain Linear today, so vision quantization is a future hook."))
    var visionBits: Int?

    @Option(name: .long, help: "Revision (branch/tag/commit) to download from HF. Default: main.")
    var revision: String = "main"

    func run() async throws {
        print("ffai \(FFAI.version) — convert")

        // ─── Resolve source ──────────────────────────────────────────
        print("resolving \(source) …")
        let locator = ModelLocator()
        let sourceDir = try await locator.resolve(
            idOrPath: source,
            revision: revision,
            progressHandler: { p in
                Task { @MainActor in
                    let frac = p.fractionCompleted
                    if frac > 0 {
                        let pct = Int(frac * 100)
                        print("  download \(pct)%", terminator: "\r")
                    }
                }
            }
        )
        print("source dir: \(sourceDir.path)")

        // ─── Compute output path ─────────────────────────────────────
        let destDir: URL
        if let out = output {
            let expanded = (out as NSString).expandingTildeInPath
            destDir = URL(fileURLWithPath: expanded, isDirectory: true)
        } else {
            destDir = defaultOutputDir(for: source, bits: bits)
        }
        print("output dir: \(destDir.path)")

        // ─── Build options ───────────────────────────────────────────
        var opts = ConvertOptions()
        opts.bits = bits
        opts.embeddingBits = embeddingBits
        opts.lmHeadBits = lmHeadBits
        opts.visionBits = visionBits

        // ─── Run conversion ──────────────────────────────────────────
        // Swift 6 strict concurrency: the progress closure is @Sendable so
        // it cannot capture a local `var` by mutation. Use a simple print
        // without a counter (the conversion is synchronous on this thread).
        let startTime = Date()
        try ConvertDriver.convert(
            sourceDir: sourceDir,
            destDir: destDir,
            options: opts,
            progress: { msg in
                print(msg)
            }
        )
        let elapsed = Date().timeIntervalSince(startTime)
        print(String(format: "\nconvert done in %.1fs", elapsed))
        print("output: \(destDir.path)")

        // ─── Optional HF upload ──────────────────────────────────────
        if let repo = uploadRepo {
            try uploadToHuggingFace(repoId: repo, directory: destDir)
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    /// Default output path: `~/.cache/ffai/converts/<safe-name>-<bits>bit`.
    /// `safe-name` is the source with "/" replaced by "--" so it stays
    /// one directory level deep and is human-readable.
    private func defaultOutputDir(for source: String, bits: Int) -> URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let cacheRoot =
            home
            .appendingPathComponent(".cache")
            .appendingPathComponent("ffai")
            .appendingPathComponent("converts")

        // For HF repo ids like "org/model", produce "org--model-4bit".
        // For local paths, use the last path component.
        let baseName: String
        if ModelLocator.isLocalPath(source) {
            baseName = URL(fileURLWithPath: source).lastPathComponent
        } else {
            baseName = source.replacingOccurrences(of: "/", with: "--")
        }
        let dirName = "\(baseName)-\(bits)bit"
        return cacheRoot.appendingPathComponent(dirName)
    }

    /// Shell out to `hf upload <repo> <dir>` for the optional upload step.
    /// The HF Python SDK (huggingface_hub) is a thin Python CLI; no Swift
    /// SDK is available in this codebase, so we use Process.
    private func uploadToHuggingFace(repoId: String, directory: URL) throws {
        print("\nuploading to \(repoId) …")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["hf", "upload", repoId, directory.path]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        let output =
            String(
                data: pipe.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8) ?? ""
        if !output.isEmpty { print(output) }

        if process.terminationStatus != 0 {
            // Non-fatal: the model was written locally even if upload fails.
            print(
                "warning: hf upload exited \(process.terminationStatus) — "
                    + "model is still at \(directory.path)")
        } else {
            print("uploaded: https://huggingface.co/\(repoId)")
        }
    }
}
