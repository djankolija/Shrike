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
    }
}
