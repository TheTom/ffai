import Testing
@testable import FFAI

@Suite("Tensor", .serialized)
struct TensorTests {
    @Test("empty allocates correct byte length")
    func empty() {
        let t = Tensor.empty(shape: [4, 8], dtype: .f32)
        #expect(t.shape == [4, 8])
        #expect(t.dtype == .f32)
        #expect(t.elementCount == 32)
        #expect(t.byteCount == 128)
        #expect(t.offset == 0)
        #expect(t.buffer.length >= 128)
    }

    @Test("reshape preserves element count and storage")
    func reshape() {
        let t = Tensor.empty(shape: [2, 6], dtype: .f32)
        let reshaped = t.reshaped(to: [3, 4])
        #expect(reshaped.shape == [3, 4])
        #expect(reshaped.elementCount == 12)
        #expect(reshaped.buffer === t.buffer)
        #expect(reshaped.offset == t.offset)
    }

    @Test("toArray / copyIn round-trip")
    func roundTrip() {
        let t = Tensor.empty(shape: [5], dtype: .f32)
        let values: [Float] = [1.5, -2.5, 0, 4.25, 100.0]
        t.copyIn(from: values)
        #expect(t.toArray(as: Float.self) == values)
    }

    @Test("zero clears bytes")
    func zero() {
        let t = Tensor.empty(shape: [3], dtype: .f32)
        t.copyIn(from: [Float(1), Float(2), Float(3)])
        t.zero()
        #expect(t.toArray(as: Float.self) == [0, 0, 0])
    }

    @Test("slicedRows updates offset and shape, shares buffer")
    func slicedRows() {
        let t = Tensor.empty(shape: [4, 3], dtype: .f32)
        let values: [Float] = (0..<12).map { Float($0) }
        t.copyIn(from: values)

        let slice = t.slicedRows(start: 1, count: 2)
        #expect(slice.shape == [2, 3])
        #expect(slice.buffer === t.buffer)
        #expect(slice.offset == 1 * 3 * MemoryLayout<Float>.size)
        #expect(slice.toArray(as: Float.self) == [3, 4, 5, 6, 7, 8])
    }
}
