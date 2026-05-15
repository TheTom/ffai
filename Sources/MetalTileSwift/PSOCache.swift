// PSOCache
//
// Lazily compiles MTLComputePipelineState objects from the metallib's
// MTLFunctions and caches them by kernel name. First-call cost is the
// PSO compilation (~100-500µs in Metal driver — no MSL parsing since the
// metallib is already pre-compiled). Subsequent calls are a hash lookup.
//
// Phase 0: keyed by kernel name only. Function-constant specialization
// keys land when we add `#[autotune]` and quantized kernels.

import Foundation
import Metal

public enum PSOCacheError: Error, CustomStringConvertible {
    case kernelNotFound(String)
    case psoCompileFailed(String, Error)

    public var description: String {
        switch self {
        case .kernelNotFound(let name):
            return "kernel '\(name)' not found in kernels.metallib"
        case .psoCompileFailed(let name, let underlying):
            return "PSO compile failed for '\(name)': \(underlying)"
        }
    }
}

public final class PSOCache: @unchecked Sendable {
    public static let shared = PSOCache(library: MetalTileLibrary.shared)

    private let library: MetalTileLibrary
    private let lock = NSLock()
    private var cache: [String: MTLComputePipelineState] = [:]

    public init(library: MetalTileLibrary) {
        self.library = library
    }

    /// Get (or build + cache) the PSO for a kernel by name.
    /// `fatalError`s on lookup or compile failure. Use `pipelineState(for:)
    /// throws -> ...` if you want to handle errors at the call site.
    public func pipelineState(for kernelName: String) -> MTLComputePipelineState {
        do {
            return try lookup(kernelName)
        } catch {
            fatalError("PSOCache.pipelineState(for: \"\(kernelName)\") failed: \(error)")
        }
    }

    public func pipelineStateThrowing(for kernelName: String) throws -> MTLComputePipelineState {
        try lookup(kernelName)
    }

    private func lookup(_ name: String) throws -> MTLComputePipelineState {
        lock.lock()
        if let cached = cache[name] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        guard let function = library.library.makeFunction(name: name) else {
            throw PSOCacheError.kernelNotFound(name)
        }
        let pso: MTLComputePipelineState
        do {
            pso = try library.device.makeComputePipelineState(function: function)
        } catch {
            throw PSOCacheError.psoCompileFailed(name, error)
        }

        lock.lock()
        cache[name] = pso
        lock.unlock()
        return pso
    }
}
