import Testing
@testable import FFAI

@Suite("Device")
struct DeviceTests {
    @Test("shared device + queue available")
    func shared() {
        let d = Device.shared
        #expect(d.mtlDevice.name.count > 0)
        // commandQueue is non-optional after construction
        _ = d.commandQueue
    }

    @Test("makeBuffer allocates requested length")
    func makeBuffer() {
        let buf = Device.shared.makeBuffer(length: 1024)
        #expect(buf.length >= 1024)
    }

    @Test("makeCommandBuffer returns a usable buffer")
    func makeCommandBuffer() {
        let cb = Device.shared.makeCommandBuffer()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)
    }
}
