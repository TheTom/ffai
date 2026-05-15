// Numeric types FFAI tensors can hold. Matches metaltile's DType subset
// we actually use for inference.

import Foundation

public enum DType: String, Sendable, Hashable, Codable {
    case f32, f16, bf16, i32, u32, i8, u8

    public var byteSize: Int {
        switch self {
        case .f32, .i32, .u32: return 4
        case .f16, .bf16: return 2
        case .i8, .u8: return 1
        }
    }

    /// Parse a SafeTensors dtype string (e.g. "F32", "F16", "BF16").
    public static func fromSafeTensors(_ s: String) -> DType? {
        switch s.uppercased() {
        case "F32": return .f32
        case "F16": return .f16
        case "BF16": return .bf16
        case "I32": return .i32
        case "U32": return .u32
        case "I8":  return .i8
        case "U8":  return .u8
        default: return nil
        }
    }

    /// Suffix used in metaltile-emit kernel names ("rms_norm_f16", etc.)
    public var kernelSuffix: String { rawValue }
}
