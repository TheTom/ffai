// Differential-debug harness for the Gemma 4 incoherent-generation bug.
// Decodes the fixed prompt token-by-token (the same path generate uses)
// and, with FFAI_INSPECT=1, dumps per-layer hidden state for the final
// prompt token so it can be diffed against the mlx-lm reference.
//
// Not part of the standard gate — guarded behind GEMMA4_DEBUG=1.

import Foundation
import Testing
@testable import FFAI

@Suite("Gemma 4 debug", .serialized)
struct Gemma4DebugTests {

    @Test("per-layer activation dump for the fixed prompt")
    func dumpActivations() async throws {
        guard ProcessInfo.processInfo.environment["GEMMA4_DEBUG"] == "1" else {
            print("Gemma4DebugTests skipped (set GEMMA4_DEBUG=1).")
            return
        }
        let modelId = "mlx-community/gemma-4-e2b-it-bf16"
        let m: Model
        do {
            m = try await ModelLoadLock.shared.loadSerially { try await Model.load(modelId) }
        } catch {
            print("Gemma 4 debug test skipped: \(error)")
            return
        }

        let prompt = "The capital of France is"
        let tokens = m.tokenizer.encode(text: prompt)
        print("FFAI prompt tokens: \(tokens)")

        // Raw table-row diagnostic: read embed_tokens_per_layer.weight[563]
        // and the main embed_tokens.weight[563] directly off the loaded
        // tensors to rule out a load-time corruption.
        if let g4 = m.engine as? Gemma4Model {
            func dumpRow(_ t: Tensor, label: String, row: Int) {
                let cols = t.shape.count == 2 ? t.shape[1] : t.elementCount
                let basePtr = t.buffer.contents().advanced(by: t.offset)
                var vals: [Float] = []
                for d in 0..<min(8, cols) {
                    let idx = row * cols + d
                    switch t.dtype {
                    case .f32:
                        vals.append(basePtr.bindMemory(to: Float.self,
                            capacity: idx + 1)[idx])
                    case .f16:
                        vals.append(halfBitsToFloatForTest(
                            basePtr.bindMemory(to: UInt16.self, capacity: idx + 1)[idx]))
                    case .bf16:
                        vals.append(bf16BitsToFloatForTest(
                            basePtr.bindMemory(to: UInt16.self, capacity: idx + 1)[idx]))
                    default: vals.append(.nan)
                    }
                }
                let head = vals.map { String(format: "%.6f", $0) }.joined(separator: ", ")
                print("[\(label)] shape=\(t.shape) dtype=\(t.dtype) row[\(row)][:8]=[\(head)]")
            }
            dumpRow(g4.embedTokens.weight, label: "embed_tokens.weight", row: 563)
            if let ple = g4.ple {
                dumpRow(ple.embed.weight, label: "embed_tokens_per_layer.weight", row: 563)

                // Directly gather token 563 through ple.embed and compare
                // with the raw row above. If they differ, the gather op is
                // the bug; if they match, the bug is downstream.
                func gatherDiag() {
                    let dev = Device.shared
                    let tb = dev.makeBuffer(length: 4)
                    var tid563 = UInt32(563)
                    memcpy(tb.contents(), &tid563, 4)
                    let tt = Tensor(buffer: tb, offset: 0, shape: [1], dtype: .u32)
                    let gcmd = dev.makeCommandBuffer()
                    let gathered = ple.embed(tt, on: gcmd)
                    gcmd.commit()
                    gcmd.waitUntilCompleted()
                    let gf = gathered.toFloatArray()
                    let gh = (0..<8).map { String(format: "%.6f", gf[$0]) }
                        .joined(separator: ", ")
                    print("[ple.embed gather(563)] shape=\(gathered.shape) [:8]=[\(gh)]")
                }
                gatherDiag()
            }
        }

        // PLE depends only on the token id (not position), so a single
        // forward of the last prompt token suffices for the PLE diff.
        let singleToken = ProcessInfo.processInfo.environment["GEMMA4_DEBUG_SINGLE"] == "1"
        let caches = m.engine.makeLayerCaches()
        var lastLogits: Tensor? = nil
        if singleToken {
            let t = tokens.last ?? 563
            print("=== single forward token id=\(t) ===")
            lastLogits = m.engine.forward(tokenId: t, position: 0, caches: caches)
        } else {
            // Decode each prompt token through the real forward path.
            for (i, t) in tokens.enumerated() {
                print("=== forward token[\(i)] id=\(t) ===")
                lastLogits = m.engine.forward(tokenId: t, position: i, caches: caches)
            }
        }
        // Argmax of the final logits.
        if let logits = lastLogits {
            let floats = logits.toFloatArray()
            var best = 0
            var bestV = -Float.infinity
            for (idx, v) in floats.enumerated() where v > bestV {
                bestV = v; best = idx
            }
            print("FFAI ARGMAX next token: \(best) value=\(bestV)")
        }
    }
}
