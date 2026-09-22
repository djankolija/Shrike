import Foundation
import Synchronization

final class InstallProgressPrinter: Sendable {
    private let printedPercent = Mutex(-1)

    func report(_ progress: ModelInstallProgress) {
        guard let line = line(for: progress) else { return }
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    func line(for progress: ModelInstallProgress) -> String? {
        switch progress {
        case .downloadingMetadata:
            return "Reading the source's metadata"
        case .planning(let downloadBytes, let outputBytes):
            return "Planned: \(Self.size(downloadBytes)) to read, \(Self.size(outputBytes)) to write"
        case .checkingDisk(let requirement):
            return "Disk: \(Self.size(requirement.requiredBytes)) needed, "
                + "\(Self.size(requirement.availableBytes)) free"
        case .reservingOutput(let bytes):
            return "Reserving \(Self.size(bytes)) for the output"
        case .copyingPayload(let reusedBytes, let downloadedThisRunBytes, let totalBytes):
            return copyLine(done: reusedBytes + downloadedThisRunBytes, total: totalBytes)
        case .hashingOutput(let relativePath):
            return "Hashing \(relativePath)"
        case .finalizing:
            return "Finalizing"
        }
    }

    private func copyLine(done: UInt64, total: UInt64) -> String? {
        let percent = total == 0 ? 100 : Int(min(done, total) * 100 / total)
        let advanced = printedPercent.withLock { printed in
            guard percent > printed else { return false }
            printed = percent
            return true
        }
        guard advanced else { return nil }
        return "Copying: \(percent)% (\(Self.size(done)) of \(Self.size(total)))"
    }

    private static func size(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes), countStyle: .file)
    }
}
