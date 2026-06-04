// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//
// Proof that Swift consumes the shared FFAI Rust engine: this links the
// libffai_ffi static library and calls the C-ABI bridge in
// rust/crates/ffai-ffi. Build + run with swift/ffi-demo/run.sh.
import Foundation

func take(_ p: UnsafeMutablePointer<CChar>?) -> String {
    guard let p else { return "<null>" }
    defer { ffai_string_free(p) }
    return String(cString: p)
}

print("FFAI Rust engine, called from Swift")
print("  version:           \(take(ffai_version()))")
print("  compiled backends: \(take(ffai_compiled_backends()))")

if let dev = ffai_open_device() {
    print("  live device:       [\(take(ffai_device_backend(dev)))] \(take(ffai_device_name(dev)))")
    ffai_close_device(dev)
} else {
    print("  live device:       none on this host")
    print("  (expected on Apple — Swift's native Metal engine is primary here;")
    print("   the same shared layer runs natively on CUDA via ffai-cuda.)")
}
