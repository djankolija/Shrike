import Foundation
import Testing
@testable import ShrikeCLICore

@Suite struct LogitsDumpTests {
    private func temporaryPath() -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("logits-\(UUID().uuidString).f16").path
    }

    private func sidecar(at path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path + ".json"))
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func clean(_ path: String) {
        try? FileManager.default.removeItem(atPath: path)
        try? FileManager.default.removeItem(atPath: path + ".json")
    }

    @Test func forcedIdsReachTheSidecarWhenGiven() throws {
        let path = temporaryPath()
        defer { clean(path) }
        let sink = try FileLogitsSink(path: path, forced: [11, 12, 13])
        try sink.finish()

        #expect(try sidecar(at: path)["forced"] as? [Int] == [11, 12, 13])
    }

    @Test func theForcedKeyIsPresentAndNullWhenOmitted() throws {
        let path = temporaryPath()
        defer { clean(path) }
        let sink = try FileLogitsSink(path: path)
        try sink.finish()

        let body = try sidecar(at: path)
        #expect(body.keys.contains("forced"))
        #expect(body["forced"] is NSNull)
    }

    @Test func finishIsIdempotent() throws {
        let path = temporaryPath()
        defer { clean(path) }
        let sink = try FileLogitsSink(path: path)
        try sink.finish()
        try sink.finish()

        #expect(try sidecar(at: path)["positions"] as? Int == 0)
    }
}
