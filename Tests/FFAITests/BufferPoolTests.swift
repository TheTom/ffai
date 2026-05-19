import Testing
@testable import FFAI

@Suite("BufferPool", .serialized)
struct BufferPoolTests {
    @Test("acquire then release reuses the same buffer")
    func reuse() {
        let pool = BufferPool(device: .shared)
        let a = pool.acquire(bytes: 256)
        pool.release(a)
        let b = pool.acquire(bytes: 256)
        #expect(b === a)
    }

    @Test("different sizes get different free lists")
    func separateSizes() {
        let pool = BufferPool(device: .shared)
        let small = pool.acquire(bytes: 64)
        let large = pool.acquire(bytes: 1024)
        pool.release(small)
        pool.release(large)
        let smallAgain = pool.acquire(bytes: 64)
        let largeAgain = pool.acquire(bytes: 1024)
        #expect(smallAgain === small)
        #expect(largeAgain === large)
    }

    @Test("acquireTensor wraps a pool buffer with the right shape/dtype")
    func acquireTensor() {
        let pool = BufferPool(device: .shared)
        let t = pool.acquireTensor(shape: [16], dtype: .f32)
        #expect(t.shape == [16])
        #expect(t.dtype == .f32)
        #expect(t.buffer.length >= 64)
    }

    @Test("releaseAll empties the freelist")
    func releaseAll() {
        let pool = BufferPool(device: .shared)
        let buf = pool.acquire(bytes: 32)
        pool.release(buf)
        pool.releaseAll()
        let fresh = pool.acquire(bytes: 32)
        #expect(fresh !== buf)
    }

    @Test("shared singleton is the same instance")
    func sharedInstance() {
        #expect(BufferPool.shared === BufferPool.shared)
    }
}
