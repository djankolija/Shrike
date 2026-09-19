import CryptoKit
import Foundation
import Shrike

enum InstrumentError: Error, CustomStringConvertible {
    case cannotCreate(String)

    var description: String {
        switch self {
        case .cannotCreate(let path): return "cannot create \(path)"
        }
    }
}

/// unchecked-invariant: one generation loop writes at a time (the single-in-flight
/// contract upstream); `finish()` reads only after the loop returns.
final class FileLogitsSink: LogitsSink, @unchecked Sendable {
    private let path: String
    private let handle: FileHandle
    private let forced: [Int32]?
    private var chosen: [Int32] = []
    private var positions = 0
    private var vocab = 0
    private var firstWriteError: Error?
    private var finished = false

    init(path: String, forced: [Int32]? = nil) throws {
        guard FileManager.default.createFile(atPath: path, contents: nil) else {
            throw InstrumentError.cannotCreate(path)
        }
        self.path = path
        self.handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        self.forced = forced
    }

    func record(position: Int, logits: UnsafeBufferPointer<Float16>) {
        guard firstWriteError == nil else { return }
        do {
            try handle.write(contentsOf: Data(buffer: logits))
            positions += 1
            vocab = logits.count
        } catch {
            firstWriteError = error
        }
    }

    func chose(position: Int, token: Int32) {
        chosen.append(token)
    }

    /// Closes the rows and writes the sidecar for however many positions landed;
    /// idempotent, and the first row write that failed is rethrown at the end.
    func finish() throws {
        guard !finished else { return }
        finished = true
        try handle.close()
        let forcedValue: Any = forced.map { $0.map(Int.init) } ?? NSNull()
        let sidecar: [String: Any] = [
            "vocab": vocab,
            "positions": positions,
            "chosen": chosen.map(Int.init),
            "forced": forcedValue,
            "binary_sha256": Self.buildHash(),
        ]
        let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path + ".json"))
        if let firstWriteError { throw firstWriteError }
    }

    /// The executable and every Metal source beside it: the kernels compile from
    /// those at run time, so a stale bundle changes the build without changing the
    /// binary.
    static func buildHash() -> String {
        guard let executable = Bundle.main.executableURL,
              let binary = try? Data(contentsOf: executable) else { return "unknown" }
        var hasher = SHA256()
        hasher.update(data: binary)
        let bundleDir = executable.deletingLastPathComponent()
        let metalFiles = (FileManager.default.enumerator(at: bundleDir, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "metal" } ?? [])
            .sorted { $0.path < $1.path }
        for file in metalFiles {
            if let source = try? Data(contentsOf: file) {
                hasher.update(data: Data(file.lastPathComponent.utf8))
                hasher.update(data: source)
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// unchecked-invariant: as `FileLogitsSink`, one generation loop writes at a
/// time and `finish()` reads only after the loop returns.
final class FileHiddenSink: HiddenSink, @unchecked Sendable {
    private let path: String
    private let handle: FileHandle
    private var positions: [Int] = []
    private var width = 0
    private var firstWriteError: Error?
    private var finished = false

    init(path: String) throws {
        guard FileManager.default.createFile(atPath: path, contents: nil) else {
            throw InstrumentError.cannotCreate(path)
        }
        self.path = path
        self.handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
    }

    func record(position: Int, hidden: UnsafeBufferPointer<Float16>) {
        guard firstWriteError == nil else { return }
        do {
            try handle.write(contentsOf: Data(buffer: hidden))
            positions.append(position)
            width = hidden.count
        } catch {
            firstWriteError = error
        }
    }

    func finish() throws {
        guard !finished else { return }
        finished = true
        try handle.close()
        let sidecar: [String: Any] = [
            "hidden": width,
            "positions": positions,
            "binary_sha256": FileLogitsSink.buildHash(),
        ]
        let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.sortedKeys])
        try data.write(to: URL(fileURLWithPath: path + ".json"))
        if let firstWriteError { throw firstWriteError }
    }
}
