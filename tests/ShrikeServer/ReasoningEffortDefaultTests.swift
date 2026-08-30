import Foundation
import Testing
@testable import Shrike
@testable import ShrikeServerCore

struct ReasoningEffortDefaultTests {
    @Test func explicitEffortAlwaysWins() {
        let resolved = ServerModelSession.defaultReasoningEffort(
            explicit: .high, dialect: .harmony, thinkingMode: .off)
        #expect(resolved.effort == .high)
        #expect(resolved.warning == nil)
    }

    @Test func thinkingOffOnHarmonyWarnsAndLowers() {
        let resolved = ServerModelSession.defaultReasoningEffort(
            explicit: nil, dialect: .harmony, thinkingMode: .off)
        #expect(resolved.effort == .low)
        #expect(resolved.warning != nil)
    }

    @Test func otherDialectsAndModesStayMedium() {
        #expect(ServerModelSession.defaultReasoningEffort(
            explicit: nil, dialect: .chatml, thinkingMode: .off).effort == .medium)
        #expect(ServerModelSession.defaultReasoningEffort(
            explicit: nil, dialect: .harmony, thinkingMode: .adaptive).effort == .medium)
        #expect(ServerModelSession.defaultReasoningEffort(
            explicit: nil, dialect: .kimi, thinkingMode: .off).effort == .medium)
    }

    @Test func explicitEffortWarnsOffHarmony() {
        let chatml = ServerModelSession.defaultReasoningEffort(
            explicit: .high, dialect: .chatml, thinkingMode: .off)
        #expect(chatml.effort == .high)
        #expect(chatml.warning != nil)

        let kimi = ServerModelSession.defaultReasoningEffort(
            explicit: .high, dialect: .kimi, thinkingMode: .off)
        #expect(kimi.effort == .high)
        #expect(kimi.warning != nil)
    }

    @Test func requireDialectSupportsThrowsWhenEffortSetOffHarmony() {
        let expected = ServerRequestError.invalid(
            message: "reasoning_effort is not supported by this model",
            param: "reasoning_effort",
            code: "unsupported_parameter")
        #expect(throws: expected) {
            try ServerModelSession.requireDialectSupports(.high, dialect: .chatml)
        }
        #expect(throws: expected) {
            try ServerModelSession.requireDialectSupports(.high, dialect: .kimi)
        }
    }

    @Test func requireDialectSupportsPassesForHarmonyOrNil() throws {
        try ServerModelSession.requireDialectSupports(.high, dialect: .harmony)
        try ServerModelSession.requireDialectSupports(nil, dialect: .chatml)
        try ServerModelSession.requireDialectSupports(nil, dialect: .kimi)
    }

    @Test func effectiveReasoningEffortPrefersRequestOverDefault() {
        #expect(ServerModelSession.effectiveReasoningEffort(
            request: .high, default: .medium) == .high)
        #expect(ServerModelSession.effectiveReasoningEffort(
            request: nil, default: .medium) == .medium)
    }
}
