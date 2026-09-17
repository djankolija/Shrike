import ShrikeAppCore
import SwiftUI

struct RunnerDiagnosticsSection: View {
    let diagnostics: AppDiagnostics?

    var body: some View {
        Section("Last run") {
            if let diagnostics {
                groupLabel("Result")
                DiagnosticRow("Settings", diagnostics.runtimeOptions.resultSummary, multiline: true)
                DiagnosticRow("Prompt tokens", diagnostics.promptTokenCount.map(String.init) ?? "unknown")
                DiagnosticRow("Output tokens", "\(diagnostics.generatedTokens)")
                DiagnosticRow("Stop", diagnostics.stopReason.rawValue)

                groupLabel("Performance")
                DiagnosticRow("Prompt prefill", MetricFormat.seconds(diagnostics.prefillSeconds))
                DiagnosticRow("First token wait", MetricFormat.seconds(diagnostics.timeToFirstTokenSeconds))
                DiagnosticRow("Request TTFT", MetricFormat.seconds(diagnostics.requestStartTimeToFirstTokenSeconds))
                DiagnosticRow("Decode duration", MetricFormat.seconds(diagnostics.decodeSeconds))
                DiagnosticRow("Decode rate", "\(MetricFormat.rate(diagnostics.tokensPerSecond)) tok/s")
                DiagnosticRow("Peak memory", MetricFormat.memory(diagnostics.peakMemoryBytes))
                DiagnosticRow("I/O / token",
                              MetricFormat.milliseconds(diagnostics.runner?.ioMillisecondsPerToken))

                if hasIssues(diagnostics) {
                    groupLabel("Issues")
                    issueRows(diagnostics)
                }

                if let prefill = diagnostics.prefill {
                    DisclosureGroup("Prefill details") {
                        VStack(spacing: 8) {
                            DiagnosticRow("Mode", "\(prefill.requestedMode.rawValue) -> \(prefill.executedMode.rawValue)")
                            DiagnosticRow("KV storage", prefill.kvStorageMode?.rawValue ?? "unknown")
                            DiagnosticRow("Completeness", prefill.chunkCompleteness.rawValue)
                            if let reason = prefill.unsupportedReason, !reason.isEmpty {
                                DiagnosticRow("Unsupported reason", reason, multiline: true)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                }
                if let runner = diagnostics.runner {
                    DisclosureGroup("Decode runner") {
                        AdvancedRunnerDiagnosticsView(runner: runner)
                    }
                }
            } else {
                Text("No runs yet")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private func issueRows(_ diagnostics: AppDiagnostics) -> some View {
        if let prefill = diagnostics.prefill {
            if prefill.requestedMode.rawValue != prefill.executedMode.rawValue {
                DiagnosticRow("Prefill mode",
                              "\(prefill.requestedMode.rawValue) -> \(prefill.executedMode.rawValue)")
            }
            if prefill.chunkCompleteness != .complete {
                DiagnosticRow("Prefill status", prefill.chunkCompleteness.rawValue)
            }
            if let reason = prefill.unsupportedReason, !reason.isEmpty {
                DiagnosticRow("Unsupported reason", reason, multiline: true)
            }
        }
    }

    private func hasIssues(_ diagnostics: AppDiagnostics) -> Bool {
        diagnostics.prefill.map {
            $0.requestedMode.rawValue != $0.executedMode.rawValue
                || $0.chunkCompleteness != .complete
                || !($0.unsupportedReason?.isEmpty ?? true)
        } ?? false
    }

    private func groupLabel(_ title: String) -> some View {
        Text(title)
            .font(.caption)
            .textCase(.uppercase)
            .foregroundStyle(.tertiary)
            .accessibilityHeading(.h3)
    }
}

private struct AdvancedRunnerDiagnosticsView: View {
    let runner: AppRunnerDiagnostics

    var body: some View {
        VStack(spacing: 8) {
            DiagnosticRow("cb1 / token", MetricFormat.milliseconds(runner.cb1MillisecondsPerToken))
            DiagnosticRow("Head / token", MetricFormat.milliseconds(runner.headMillisecondsPerToken))
        }
        .frame(maxWidth: .infinity)
    }
}

private struct DiagnosticRow: View {
    let label: String
    let value: String
    let multiline: Bool

    init(_ label: String, _ value: String, multiline: Bool = false) {
        self.label = label
        self.value = value
        self.multiline = multiline
    }

    var body: some View {
        LabeledContent(label) {
            Text(value)
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .multilineTextAlignment(.trailing)
                .lineLimit(multiline ? nil : 2)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
