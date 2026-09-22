import Foundation
import Testing
@testable import ShrikeRepackCore

@Suite struct DiskSpaceCheckerTests {
    @Test func theRequirementIsTheBytesPlusTheReserve() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("shrike-space-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("model.gturbo", isDirectory: true)

        let required = try DiskSpaceChecker.requireAvailable(path: target.path,
                                                             bytes: 100,
                                                             reserveBytes: 20)
        #expect(required.requiredBytes == 120)
    }

    @Test func insufficientCheckReportsRequiredAndAvailableBytes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("shrike-space-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("model.gturbo", isDirectory: true)
        let required = UInt64.max

        #expect {
            _ = try DiskSpaceChecker.requireAvailable(path: target.path,
                                                      bytes: required,
                                                      reserveBytes: 0)
        } throws: { error in
            guard case RepackError.diskSpaceInsufficient(_, let reportedRequired, let actual) = error else {
                return false
            }
            return reportedRequired == required && actual < required
        }
    }
}
