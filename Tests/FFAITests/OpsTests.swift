import Foundation
import Metal
import Testing
@testable import FFAI

@Suite("Ops")
struct OpsTests {
    func runAndWait(_ block: (MTLCommandBuffer) -> Void) {
        let cb = Device.shared.makeCommandBuffer()
        block(cb)
        cb.commit()
        cb.waitUntilCompleted()
    }

    @Test("add f32 — c[i] = a[i] + b[i]")
    func addF32() {
        let a = Tensor.empty(shape: [5], dtype: .f32)
        let b = Tensor.empty(shape: [5], dtype: .f32)
        a.copyIn(from: [Float(1), 2, 3, 4, 5])
        b.copyIn(from: [Float(10), 20, 30, 40, 50])
        var out: Tensor!
        runAndWait { cb in out = Ops.add(a, b, on: cb) }
        #expect(out.toArray(as: Float.self) == [11, 22, 33, 44, 55])
    }

    @Test("mul f32 — c[i] = a[i] * b[i]")
    func mulF32() {
        let a = Tensor.empty(shape: [4], dtype: .f32)
        let b = Tensor.empty(shape: [4], dtype: .f32)
        a.copyIn(from: [Float(1), 2, 3, 4])
        b.copyIn(from: [Float(5), 6, 7, 8])
        var out: Tensor!
        runAndWait { cb in out = Ops.mul(a, b, on: cb) }
        #expect(out.toArray(as: Float.self) == [5, 12, 21, 32])
    }

    @Test("silu f32 — out[i] = x / (1 + exp(-x))")
    func siluF32() {
        let x = Tensor.empty(shape: [4], dtype: .f32)
        x.copyIn(from: [Float(0), 1, -1, 2])
        var out: Tensor!
        runAndWait { cb in out = Ops.silu(x, on: cb) }
        let result = out.toArray(as: Float.self)
        // silu(0) = 0
        #expect(abs(result[0]) < 1e-5)
        // silu(1) ≈ 0.7311
        #expect(abs(result[1] - Float(1.0 / (1.0 + exp(-1.0)))) < 1e-3)
        // silu(-1) ≈ -0.2689
        #expect(abs(result[2] - Float(-1.0 / (1.0 + exp(1.0)))) < 1e-3)
        // silu(2) ≈ 1.7616
        #expect(abs(result[3] - Float(2.0 / (1.0 + exp(-2.0)))) < 1e-3)
    }

    @Test("gather f32 — picks the right rows")
    func gatherF32() {
        // table[3, 2] = [[10,11], [20,21], [30,31]]
        let table = Tensor.empty(shape: [3, 2], dtype: .f32)
        table.copyIn(from: [Float(10), 11, 20, 21, 30, 31])
        let ids = Tensor.empty(shape: [2], dtype: .u32)
        ids.copyIn(from: [UInt32(2), 0])
        var out: Tensor!
        runAndWait { cb in out = Ops.gather(table: table, tokenIds: ids, on: cb) }
        #expect(out.shape == [2, 2])
        #expect(out.toArray(as: Float.self) == [30, 31, 10, 11])
    }

    @Test("gemv f32 — out[i] = sum_j W[i,j] * x[j]")
    func gemvF32() {
        // W [3, 2] = [[1,2], [3,4], [5,6]]
        let w = Tensor.empty(shape: [3, 2], dtype: .f32)
        w.copyIn(from: [Float(1), 2, 3, 4, 5, 6])
        let x = Tensor.empty(shape: [2], dtype: .f32)
        x.copyIn(from: [Float(7), 8])
        // expected: [1*7+2*8, 3*7+4*8, 5*7+6*8] = [23, 53, 83]
        var out: Tensor!
        runAndWait { cb in out = Ops.gemv(weight: w, input: x, on: cb) }
        #expect(out.toArray(as: Float.self) == [23, 53, 83])
    }

    @Test("rmsNorm f32 — y = x / rms(x) * weight")
    func rmsNormF32() {
        let x = Tensor.empty(shape: [4], dtype: .f32)
        x.copyIn(from: [Float(1), 2, 3, 4])
        let weight = Tensor.empty(shape: [4], dtype: .f32)
        weight.copyIn(from: [Float(1), 1, 1, 1])
        var out: Tensor!
        runAndWait { cb in out = Ops.rmsNorm(x, weight: weight, eps: 1e-6, on: cb) }
        let result = out.toArray(as: Float.self)
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // expected ≈ x / 2.7386 = [0.365, 0.730, 1.095, 1.461]
        let expectedRms = Float((30.0 / 4.0).squareRoot())
        for i in 0..<4 {
            let expected = Float(i + 1) / expectedRms
            #expect(abs(result[i] - expected) < 1e-3,
                    "i=\(i) got \(result[i]) expected \(expected)")
        }
    }

    @Test("rope f32 at position 0 is identity (cos=1, sin=0)")
    func ropePos0Identity() {
        let qk = Tensor.empty(shape: [1, 4], dtype: .f32)
        qk.copyIn(from: [Float(1), 2, 3, 4])
        var out: Tensor!
        runAndWait { cb in
            out = Ops.rope(qk, position: 0, headDim: 4, thetaBase: 10000, on: cb)
        }
        let r = out.toArray(as: Float.self)
        // theta = 0 → cos = 1, sin = 0 → identity
        for i in 0..<4 {
            #expect(abs(r[i] - Float(i + 1)) < 1e-4, "i=\(i) got \(r[i])")
        }
    }

    @Test("rope f32 at position 1, head_dim=4, theta_base=10000")
    func ropePos1() {
        let qk = Tensor.empty(shape: [1, 4], dtype: .f32)
        qk.copyIn(from: [Float(1), 0, 0, 1])
        var out: Tensor!
        runAndWait { cb in
            out = Ops.rope(qk, position: 1, headDim: 4, thetaBase: 10000, on: cb)
        }
        // i_pair=0: inv_freq = 1, theta = 1, cos≈0.5403, sin≈0.8415
        //   x[0] * cos - x[2] * sin = 1*0.5403 - 0*0.8415 = 0.5403
        //   x[0] * sin + x[2] * cos = 1*0.8415 + 0*0.5403 = 0.8415
        // i_pair=1: inv_freq = 1/sqrt(10000) = 0.01, theta = 0.01, cos≈1, sin≈0.01
        //   x[1]*cos - x[3]*sin = 0*1 - 1*0.01 = -0.01
        //   x[1]*sin + x[3]*cos = 0*0.01 + 1*1 = 1
        let r = out.toArray(as: Float.self)
        #expect(abs(r[0] - 0.5403) < 1e-3)   // index 0
        #expect(abs(r[1] - (-0.01)) < 5e-3)  // index 1 (looser tol for f32 trig)
        #expect(abs(r[2] - 0.8415) < 1e-3)   // index 0 + half_dim
        #expect(abs(r[3] - 1.0) < 1e-3)      // index 1 + half_dim
    }

    @Test("sdpaDecode f32 — single position attends to itself")
    func sdpaSinglePosition() {
        // 1 q-head, 1 kv-head, head_dim=4, n_kv=1
        let q = Tensor.empty(shape: [1, 4], dtype: .f32)
        let k = Tensor.empty(shape: [1, 4, 4], dtype: .f32)  // [n_kv_heads, kv_stride=4, head_dim]
        let v = Tensor.empty(shape: [1, 4, 4], dtype: .f32)
        q.copyIn(from: [Float(1), 0, 0, 0])
        // Only the first position is filled
        var kData = [Float](repeating: 0, count: 16)
        kData[0] = 1; kData[1] = 0; kData[2] = 0; kData[3] = 0
        k.copyIn(from: kData)
        var vData = [Float](repeating: 0, count: 16)
        vData[0] = 7; vData[1] = 8; vData[2] = 9; vData[3] = 10
        v.copyIn(from: vData)

        var out: Tensor!
        runAndWait { cb in
            out = Ops.sdpaDecode(q: q, k: k, v: v,
                                 nQHeads: 1, nKVHeads: 1, headDim: 4,
                                 nKV: 1, kvStride: 4, scale: 1.0, on: cb)
        }
        // Single KV → softmax([single_score]) = 1 → output = v[0]
        #expect(out.toArray(as: Float.self) == [7, 8, 9, 10])
    }
}
