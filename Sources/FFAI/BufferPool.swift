// BufferPool — simple reusable MTLBuffer allocator for activation
// tensors. Phase 2 implementation: per-byte-size LIFO. No fancy
// fragmentation handling, no sub-allocation. Activations are small and
// short-lived; this is good enough until profiles say otherwise.

import Foundation
import Metal

public final class BufferPool: @unchecked Sendable {
    public let device: Device
    private let lock = NSLock()
    private var freelists: [Int: [MTLBuffer]] = [:]   // bytes → free buffers

    public static let shared = BufferPool(device: .shared)

    public init(device: Device) {
        self.device = device
    }

    /// Borrow a buffer of at least `bytes`. Caller MUST return it via
    /// `release(_:)` when done, or memory accumulates.
    public func acquire(bytes: Int) -> MTLBuffer {
        lock.lock()
        if var pool = freelists[bytes], let buf = pool.popLast() {
            freelists[bytes] = pool
            lock.unlock()
            return buf
        }
        lock.unlock()
        return device.makeBuffer(length: bytes)
    }

    /// Return a buffer to the pool. Don't use it after this call.
    public func release(_ buffer: MTLBuffer) {
        let bytes = buffer.length
        lock.lock()
        freelists[bytes, default: []].append(buffer)
        lock.unlock()
    }

    /// Allocate a Tensor from the pool. NOTE: caller is responsible for
    /// returning `tensor.buffer` to the pool when done. For typical use
    /// inside a single forward pass, drop the entire pool's buffers at
    /// end of forward pass with `releaseAll()`.
    public func acquireTensor(shape: [Int], dtype: DType) -> Tensor {
        let count = shape.reduce(1, *)
        let bytes = count * dtype.byteSize
        let buf = acquire(bytes: bytes)
        return Tensor(buffer: buf, offset: 0, shape: shape, dtype: dtype)
    }

    /// Drop everything. For Phase 2 we just reallocate freely; caller
    /// invokes this between forward passes if memory is a concern.
    public func releaseAll() {
        lock.lock()
        freelists.removeAll()
        lock.unlock()
    }
}
