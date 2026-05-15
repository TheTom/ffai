// Diagnostic — load the real 4-bit Qwen3 checkpoint, dequantize one
// embedding row two ways (kernel + CPU), and compare.

import Foundation
import Testing
@testable import FFAI

@Suite("Quantized 4-bit diagnostic", .serialized)
struct Quantized4bitDiagnostic {
    @Test("dequant gather of embed_tokens row 0 matches a Swift CPU reference")
    func dequantGatherMatchesCPU() async throws {
        let dir: URL
        do {
            dir = try await ModelLocator().resolve(
                idOrPath: "mlx-community/Qwen3-4B-4bit"
            )
        } catch {
            print("4-bit diagnostic skipped (download failed): \(error)")
            return
        }
        let cfg = try ModelConfig.load(from: dir)
        let bundle = try SafeTensorsBundle(directory: dir)
        guard let q = cfg.quantization else {
            Issue.record("expected quantization block in config")
            return
        }
        let hidden = cfg.hiddenSize ?? 2560
        let groupSize = q.groupSize

        let weight = try bundle.tensor(named: "model.embed_tokens.weight")
        let scales = try bundle.tensor(named: "model.embed_tokens.scales")
        let biases = try bundle.tensor(named: "model.embed_tokens.biases")

        // Use the safetensors-backed tensors directly (no copy)
        let ids = Tensor.empty(shape: [1], dtype: .u32)
        ids.copyIn(from: [UInt32(0)])
        let cb = Device.shared.makeCommandBuffer()
        let kernelOut = Ops.dequantGatherInt4(
            weight: weight, scales: scales, biases: biases,
            tokenIds: ids, hidden: hidden, groupSize: groupSize, on: cb
        )
        cb.commit()
        await cb.completed()
        let kernelBits = kernelOut.toArray(as: UInt16.self)
        let kernelF = kernelBits.map { Float(bitPattern: UInt32($0) << 16) }

        // CPU dequantize row 0
        let weightU32 = weight.toArray(as: UInt32.self)
        let scalesU16 = scales.toArray(as: UInt16.self)
        let biasesU16 = biases.toArray(as: UInt16.self)
        func bf(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

        let packsPerRow = hidden / 8
        let groupsPerRow = hidden / groupSize
        var cpu = [Float](repeating: 0, count: hidden)
        for d in 0..<hidden {
            let g = d / groupSize
            let pack = weightU32[0 * packsPerRow + d / 8]
            let nibble = d & 7
            let qv = Float((pack >> (4 * nibble)) & 0xF)
            let scaleV = bf(scalesU16[0 * groupsPerRow + g])
            let biasV  = bf(biasesU16[0 * groupsPerRow + g])
            cpu[d] = qv * scaleV + biasV
        }

        // Compare
        print("kernel[:8] = \(kernelF.prefix(8).map { String(format: "%.4e", $0) }.joined(separator: " "))")
        print("cpu[:8]    = \(cpu.prefix(8).map { String(format: "%.4e", $0) }.joined(separator: " "))")
        print("scales row 0 first 4: \(scalesU16.prefix(4).map { bf($0) })")

        // Identify matching positions to see the stride pattern
        var matches: [Int] = []
        for d in 0..<hidden {
            let tol: Float = max(abs(cpu[d]) * 0.05, 1e-2)
            if kernelF[d].isFinite && abs(kernelF[d] - cpu[d]) < tol {
                matches.append(d)
            }
        }
        print("matching positions (\(matches.count) total): \(matches.prefix(20))…")
        if matches.count > 0 {
            print("first match: kernel=\(kernelF[matches[0]]) cpu=\(cpu[matches[0]])")
        }
        // Print kernel output at strided positions
        let strides = [0, 64, 128, 256, 512, 1024]
        for s in strides where s < hidden {
            print("kernel[\(s)]=\(kernelF[s])  cpu[\(s)]=\(cpu[s])")
        }
        #expect(matches.count == hidden, "\(hidden - matches.count) mismatches out of \(hidden)")
    }
}
