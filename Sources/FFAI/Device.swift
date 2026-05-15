// Device — singleton wrapper over the system MTLDevice + a default
// command queue. Exposes a single Sendable handle that all of FFAI uses
// to allocate buffers and submit work.

import Foundation
import Metal
import MetalTileSwift

public final class Device: @unchecked Sendable {
    public let mtlDevice: MTLDevice
    public let commandQueue: MTLCommandQueue

    public static let shared: Device = {
        // Reuse the same MTLDevice + queue MetalTileSwift uses, so PSOs
        // and buffers are guaranteed compatible.
        let lib = MetalTileLibrary.shared
        return Device(mtlDevice: lib.device, commandQueue: lib.commandQueue)
    }()

    public init(mtlDevice: MTLDevice, commandQueue: MTLCommandQueue) {
        self.mtlDevice = mtlDevice
        self.commandQueue = commandQueue
    }

    /// Allocate a fresh shared-storage MTLBuffer of the given byte length.
    public func makeBuffer(length: Int) -> MTLBuffer {
        guard let buf = mtlDevice.makeBuffer(length: length, options: .storageModeShared) else {
            fatalError("Device.makeBuffer(length: \(length)) returned nil")
        }
        return buf
    }

    /// Make a new MTLCommandBuffer.
    public func makeCommandBuffer() -> MTLCommandBuffer {
        guard let cb = commandQueue.makeCommandBuffer() else {
            fatalError("Device.makeCommandBuffer() returned nil")
        }
        return cb
    }
}
