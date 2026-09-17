import Foundation
import Metal
import Testing

@testable import Shrike

@Suite struct TokenCommandTests {
    private final class EncoderInfo: NSObject, MTLCommandBufferEncoderInfo {
        let label: String
        let debugSignposts: [String] = []
        let errorState: MTLCommandEncoderErrorState

        init(label: String, errorState: MTLCommandEncoderErrorState) {
            self.label = label
            self.errorState = errorState
        }
    }

    @Test func aCommandErrorNamesTheFaultedEncoderAndCountsTheAffected() {
        let infos: [EncoderInfo] = [
            EncoderInfo(label: "layer 11 attention", errorState: .completed),
            EncoderInfo(label: "layer 12 attention", errorState: .completed),
            EncoderInfo(label: "layer 12 fixup", errorState: .faulted),
            EncoderInfo(label: "layer 13 attention", errorState: .affected),
            EncoderInfo(label: "", errorState: .affected),
        ]
        let error = NSError(domain: MTLCommandBufferErrorDomain, code: 1,
                            userInfo: [MTLCommandBufferEncoderInfoErrorKey: infos])
        let text = RealForwardRunner.describeCommandBufferError(error)
        #expect(text.contains("faulted: layer 12 fixup"))
        #expect(text.contains("affected encoders: 2"))
    }

    @Test func aCommandErrorWithoutEncoderInfoIsDescribedAsIs() {
        let error = NSError(domain: MTLCommandBufferErrorDomain, code: 7, userInfo: [:])
        let text = RealForwardRunner.describeCommandBufferError(error)
        #expect(text.contains("code=7") || text.contains("Code=7"))
        #expect(RealForwardRunner.describeCommandBufferError(nil) == "no error recorded")
    }

    @Test func aWaitOnACommandThatNeverCompletesEndsAtTheDeadlineNamingTheLayer() throws {
        let context = try MetalContext()
        let command = try #require(context.queue.makeCommandBuffer())
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        #expect(throws: ModelError.self) {
            try RealForwardRunner.awaitCompletion(of: command, deadlineNanos: 50_000_000,
                                                  naming: "layer 7's word")
        }
        let elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
        #expect(elapsed >= 50_000_000 && elapsed < 2_000_000_000)
        do {
            try RealForwardRunner.awaitCompletion(of: command, deadlineNanos: 1_000_000,
                                                  naming: "layer 7's word")
        } catch let error as ModelError {
            #expect(String(describing: error).contains("layer 7's word"))
        }
    }

    @Test func aWaitOnACompletedCommandReturnsAtOnce() throws {
        let context = try MetalContext()
        let command = try #require(context.queue.makeCommandBuffer())
        command.commit()
        command.waitUntilCompleted()
        try RealForwardRunner.awaitCompletion(of: command, deadlineNanos: 10_000_000_000,
                                              naming: "the boundary")
    }
}
