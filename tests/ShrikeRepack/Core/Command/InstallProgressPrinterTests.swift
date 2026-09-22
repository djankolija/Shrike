import Testing

@testable import ShrikeRepackCore

@Suite struct InstallProgressPrinterTests {
    private func copy(_ done: UInt64, of total: UInt64 = 1_000) -> ModelInstallProgress {
        .copyingPayload(reusedBytes: 0, downloadedThisRunBytes: done, totalBytes: total)
    }

    @Test func everyPhaseHasALine() {
        let printer = InstallProgressPrinter()
        let phases: [ModelInstallProgress] = [
            .downloadingMetadata,
            .planning(downloadBytes: 1, outputBytes: 1),
            .checkingDisk(DiskSpaceRequirement(path: "/", requiredBytes: 1, availableBytes: 2)),
            .reservingOutput(bytes: 1),
            .hashingOutput("packed_experts/layer_00.bin"),
            .finalizing,
        ]
        for phase in phases {
            #expect(printer.line(for: phase) != nil)
        }
        #expect(printer.line(for: .hashingOutput("manifest.json")) == "Hashing manifest.json")
    }

    @Test func theCopyPrintsOncePerWholePercent() {
        let printer = InstallProgressPrinter()
        let printed = [0, 4, 10, 15, 19, 20, 1_000].map { printer.line(for: copy($0)) != nil }
        #expect(printed == [true, false, true, false, false, true, true])
    }

    @Test func aResumedCopyStartsAtWhatWasReused() {
        let printer = InstallProgressPrinter()
        let line = printer.line(for: .copyingPayload(reusedBytes: 400,
                                                     downloadedThisRunBytes: 0,
                                                     totalBytes: 1_000))
        #expect(line?.hasPrefix("Copying: 40%") == true)
    }
}
