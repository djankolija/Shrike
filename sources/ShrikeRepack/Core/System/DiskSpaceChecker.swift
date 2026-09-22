import Foundation
import Darwin

public struct DiskSpaceRequirement: Equatable, Sendable {
    public let path: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(path: String, requiredBytes: UInt64, availableBytes: UInt64) {
        self.path = path
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }
}

public enum DiskSpaceChecker {
    public static func requireAvailable(path: String,
                                        bytes: UInt64,
                                        reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) throws -> DiskSpaceRequirement {
        try Posix.mkdirP(path)
        let result = try requirement(path: path, bytes: bytes, reserveBytes: reserveBytes)
        guard result.canInstall else {
            throw RepackError.diskSpaceInsufficient(path: path,
                                                    required: result.requiredBytes,
                                                    available: result.availableBytes)
        }
        return result
    }

    private static func requirement(path: String,
                                    bytes: UInt64,
                                    reserveBytes: UInt64) throws -> DiskSpaceRequirement {
        var st = statfs()
        if statfs(path, &st) != 0 {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        let available = UInt64(st.f_bavail) * UInt64(st.f_bsize)
        let sum = bytes.addingReportingOverflow(reserveBytes)
        guard !sum.overflow else {
            throw RepackError.configurationInvalid(
                detail: "disk-space requirement overflows UInt64")
        }
        let required = sum.partialValue
        return DiskSpaceRequirement(path: path,
                                    requiredBytes: required,
                                    availableBytes: available)
    }
}
