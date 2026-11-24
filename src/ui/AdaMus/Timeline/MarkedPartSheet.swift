// A part of a session, marked on its timeline with Shift and a drag: the program cuts it out of
// the data of the session as one JSON document (the session, a summary of the part, the blocks,
// every step of the detector, the music, the electrodes, the log, the events, the
// configuration, the files), to be read here or copied.
import AppKit
import SwiftUI

struct MarkedPartSheet: View {
    let part: MarkedPart
    let session: String
    /// The part cut out of the data by the program.
    let read: () async throws -> String
    @Environment(\.dismiss) private var dismiss
    @State private var excerpt: String?
    @State private var problem: String?

    var body: some View {
        Form {
            Section {
                LabeledContent("Session", value: session)
                LabeledContent("Part", value: Self.span(of: part))
            }
            Section("Data") {
                if let excerpt {
                    Text(excerpt)
                        .font(.callout.monospaced())
                        .textSelection(.enabled)
                } else if let problem {
                    Label(problem, systemImage: "xmark.octagon").foregroundStyle(.red)
                } else {
                    ProgressView()
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: Metrics.markedPartSheet.width, height: Metrics.markedPartSheet.height)
        .navigationTitle("Part of the Session")
        .task {
            do {
                excerpt = try await read()
            } catch {
                problem = error.localizedDescription
            }
        }
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Close") { dismiss() }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(excerpt ?? "", forType: .string)
                    dismiss()
                }
                .disabled(excerpt == nil)
            }
        }
    }

    /// From when to when the part lies, as the timeline writes times, and how long it is.
    static func span(of part: MarkedPart) -> String {
        let length = (part.endS - part.startS).formatted(.number.precision(.fractionLength(1)))
        return "\(ValueFormatting.clock(part.startS, tenths: true)) – "
            + "\(ValueFormatting.clock(part.endS, tenths: true)) (\(length) s)"
    }
}
