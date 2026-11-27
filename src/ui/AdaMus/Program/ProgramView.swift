// The window as the program starts: the checks as they run, what failed and how to fix it, or
// why the program ended; then the sessions.
import SwiftUI

struct ProgramView: View {
    @Environment(Program.self) private var program

    var body: some View {
        switch program.state {
        case .notInProject(let folder):
            ContentUnavailableView {
                Label("AdaMus Is Not in Its Project Folder", systemImage: "folder.badge.questionmark")
            } description: {
                Text(
                    "AdaMus runs the program of the folder it lies in, and \(folder.path(percentEncoded: false)) holds none. Put AdaMus back into its project folder; to open it from elsewhere, make an alias of it."
                )
            }
        case .starting(let checks) where checks.isEmpty:
            ProgressView("Starting AdaMus").frame(maxWidth: .infinity, maxHeight: .infinity)  // before its first word
        case .starting(let checks):
            Checklist(checks: checks, failed: false)
        case .failed(let checks):
            Checklist(checks: checks, failed: true)
        case .ended(let said):
            Ended(said: said)
        case .ready(let api):
            if let library = program.library {  // made with the program's API, as it became ready
                WorkspaceView(api: api, library: library)
            }
        }
    }
}

/// The checks of the start, each with what it found; once one has failed, how to fix it.
struct Checklist: View {
    @Environment(Program.self) private var program
    let checks: [Check]
    let failed: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(failed ? "AdaMus Could Not Start" : "Starting AdaMus")
                .font(.title2.weight(.semibold))
                .padding(.horizontal, Metrics.Spacing.page)
                .padding(.top, Metrics.Spacing.page)
            checklist
        }
        .frame(maxWidth: Metrics.checklistWidth)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var checklist: some View {
        Form {
            Section {
                ForEach(checks) { check in
                    CheckRow(check: check)
                }
            } footer: {
                if failed {
                    HStack {
                        Text("Fix what is named above, then try again.")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Try Again") { program.start() }
                            .keyboardShortcut(.defaultAction)
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}

/// The program ended on its own: the last lines it printed, to be read and copied, and a way to
/// start it again.
struct Ended: View {
    @Environment(Program.self) private var program
    let said: String

    var body: some View {
        VStack(spacing: Metrics.Spacing.large) {
            ContentUnavailableView {
                Label("AdaMus Stopped", systemImage: "exclamationmark.triangle")
            } description: {
                Text("The program behind the window ended. The last lines it printed say why.")
            } actions: {
                Button("Start Again") { program.start() }
            }
            .fixedSize(horizontal: false, vertical: true)
            ScrollView {
                Text(said)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(.quinary, in: RoundedRectangle(cornerRadius: Metrics.cornerRadius))
            .frame(maxWidth: Metrics.lastWords.width, maxHeight: Metrics.lastWords.height)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct CheckRow: View {
    let check: Check

    var body: some View {
        LabeledContent {
            if !check.detail.isEmpty {
                Text(check.detail)
                    .foregroundStyle(check.status == .failed ? .primary : .secondary)
                    .multilineTextAlignment(.trailing)
                    .textSelection(.enabled)
            }
        } label: {
            Label {
                VStack(alignment: .leading, spacing: Metrics.Spacing.tight) {
                    Text(check.name)
                    if let fix = check.fix {
                        Text(fix).font(.callout).foregroundStyle(.secondary)
                    }
                }
            } icon: {
                status
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityValue(check.status.spoken)
    }

    @ViewBuilder private var status: some View {
        switch check.status {
        case .pending:
            Image(systemName: "circle").foregroundStyle(.tertiary)
        case .running:
            ProgressView().controlSize(.small)
        case .passed:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        case .skipped:
            Image(systemName: "minus.circle").foregroundStyle(.secondary)
        }
    }
}

extension Check.Status {
    /// The state of a check, as VoiceOver says it.
    var spoken: String {
        switch self {
        case .pending: "Waiting"
        case .running: "Checking"
        case .passed: "Passed"
        case .failed: "Failed"
        case .skipped: "Skipped, as a check it needs did not pass"
        }
    }
}
