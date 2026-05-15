// MetalTileLibrary
//
// Loads kernels.metallib once at process startup and exposes the underlying
// MTLLibrary + a default MTLDevice and MTLCommandQueue. Designed to be a
// process-wide singleton (`MetalTileLibrary.shared`).
//
// kernels.metallib + manifest.json are produced at build time by
// metaltile-emit. See planning/architecture.md §1.

import Foundation
import Metal

public enum MetalTileLibraryError: Error, CustomStringConvertible {
    case noDefaultDevice
    case noCommandQueue
    case metallibNotFound(URL)
    case metallibLoadFailed(URL, Error)

    public var description: String {
        switch self {
        case .noDefaultDevice:
            return "MTLCreateSystemDefaultDevice() returned nil"
        case .noCommandQueue:
            return "MTLDevice.makeCommandQueue() returned nil"
        case .metallibNotFound(let url):
            return "kernels.metallib not found at \(url.path)"
        case .metallibLoadFailed(let url, let underlying):
            return "Failed to load \(url.path): \(underlying)"
        }
    }
}

public final class MetalTileLibrary: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    public let metallibURL: URL

    /// Process-wide singleton. Lazily initialized; throws on first access if
    /// the system has no default Metal device or the metallib can't be loaded.
    public static let shared: MetalTileLibrary = {
        do {
            return try MetalTileLibrary()
        } catch {
            fatalError("MetalTileLibrary.shared init failed: \(error)")
        }
    }()

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalTileLibraryError.noDefaultDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalTileLibraryError.noCommandQueue
        }
        let url = try Self.locateMetallib()
        do {
            let library = try device.makeLibrary(URL: url)
            self.device = device
            self.commandQueue = queue
            self.library = library
            self.metallibURL = url
        } catch {
            throw MetalTileLibraryError.metallibLoadFailed(url, error)
        }
    }

    /// Find kernels.metallib in the SPM resource bundle.
    private static func locateMetallib() throws -> URL {
        if let url = Bundle.module.url(
            forResource: "kernels",
            withExtension: "metallib",
            subdirectory: "Resources"
        ) {
            return url
        }
        // Fallback: SPM may flatten the Resources/ folder.
        if let url = Bundle.module.url(forResource: "kernels", withExtension: "metallib") {
            return url
        }
        let fallback = Bundle.module.bundleURL.appendingPathComponent(
            "Resources/kernels.metallib"
        )
        throw MetalTileLibraryError.metallibNotFound(fallback)
    }
}
