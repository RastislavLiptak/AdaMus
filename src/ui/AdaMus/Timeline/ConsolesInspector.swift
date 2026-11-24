// The logs of the session in the inspector beside the timeline, one at a time, in columns: when
// a line came, who wrote it and what it says, a warning or an error marked. A log shows the lines
// that came by the playhead, so that it fills as the session is played back, as a terminal
// does: it opens at its newest line and stays with the newest while it is scrolled to the end;
// scrolled up, even a little, it stays where it is. A line is copied from its menu, which does
// not touch the playback (its text is not selectable: the menu of selectable text would stand
// in for it). A Table cannot follow: it tells nothing of where it is scrolled, and takes no
// scroll anchor.
import AppKit
import SwiftUI

struct ConsolesInspector: View {
    let timeline: TimelineModel
    @State private var shown = 0

    var body: some View {
        let consoles = timeline.timeline.consoles
        VStack(spacing: 0) {
            Picker("Log", selection: $shown) {
                ForEach(Array(consoles.enumerated()), id: \.offset) { index, console in Text(console.title).tag(index) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding()
            if consoles.indices.contains(shown) {
                let lines = consoles[shown].lines
                LogTable(lines: lines, heard: Self.heard(lines, by: timeline.playhead))
                    .equatable()
                    .id(shown)  // another log: opened anew, at its newest line
            }
        }
    }

    /// How many lines of a log came by ``time``: a line at it has come. The program sends the
    /// lines in the order of their times.
    nonisolated static func heard(_ lines: [ConsoleLine], by time: Double) -> Int {
        lines.partitioningIndex { $0.timeS > time }
    }

    struct Line: Identifiable {
        let id: Int
        let line: ConsoleLine

        var symbol: (name: String, colour: Color)? {
            switch line.severity {
            case .warning: ("exclamationmark.triangle.fill", .yellow)
            case .error: ("xmark.octagon.fill", .red)
            case .debug, .info: nil
            }
        }

        var copied: String {
            [ValueFormatting.clock(line.timeS, tenths: true), line.source, line.text]
                .compactMap { $0 }.joined(separator: "\t")
        }
    }
}

/// The first ``heard`` lines of a log; drawn again only when that many changes, not as the
/// playhead moves within the same line.
struct LogTable: View, Equatable {
    let lines: [ConsoleLine]
    let heard: Int
    @State private var following = true
    @State private var position = ScrollPosition(edge: .bottom)

    nonisolated static func == (one: LogTable, other: LogTable) -> Bool {
        one.heard == other.heard && one.lines == other.lines
    }

    var body: some View {
        let rows = lines.prefix(heard).enumerated().map { ConsolesInspector.Line(id: $0.offset, line: $0.element) }
        VStack(spacing: 0) {
            LogRow.header
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(rows) { LogRow(row: $0) }
                }
            }
            .scrollPosition($position)
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .onScrollPhaseChange { old, new, context in
                let geometry = context.geometry
                let atEnd = Self.isAtEnd(
                    offset: geometry.contentOffset.y, shown: geometry.containerSize.height,
                    of: geometry.contentSize.height)
                following = Self.follows(following, from: old, to: new, atEnd: atEnd)
            }
            .onChange(of: heard) {
                // to the edge, not to the last line: the lines below the view are of heights only
                // guessed, and a scroll to the last of them may stop short of the end
                if following { position.scrollTo(edge: .bottom) }
            }
        }
    }

    /// Whether a scroll view shows the end of what it holds, as far as a point can be told.
    nonisolated static func isAtEnd(offset: CGFloat, shown: CGFloat, of height: CGFloat) -> Bool {
        offset + shown >= height - Metrics.hairline
    }

    /// Whether the log follows its newest lines. Only a person scrolling changes it: while they
    /// scroll, it does not; once they let go, it follows if they left it at its end. The log's
    /// own scrolls to its end, and new lines below the view, change nothing.
    nonisolated static func follows(_ following: Bool, from old: ScrollPhase, to new: ScrollPhase, atEnd: Bool) -> Bool
    {
        switch new {
        case .interacting, .decelerating: false
        case .idle where old != .idle: atEnd
        default: following
        }
    }
}

/// A line of a log in its columns, and the names of the columns over them.
struct LogRow: View {
    let row: ConsolesInspector.Line

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Metrics.Spacing.small) {
            Text(ValueFormatting.clock(row.line.timeS, tenths: true))
                .monospacedDigit()
                .frame(width: Metrics.logTime, alignment: .leading)
            Text(row.line.source ?? "")
                .foregroundStyle(.secondary)
                .frame(width: Metrics.logSource, alignment: .leading)
            HStack(alignment: .firstTextBaseline, spacing: Metrics.Spacing.tight) {
                if let symbol = row.symbol {
                    Image(systemName: symbol.name).foregroundStyle(symbol.colour)
                        .accessibilityLabel(row.line.severity.rawValue)
                }
                Text(row.line.text).font(.callout)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal)
        .padding(.vertical, Metrics.Spacing.tight)
        .contentShape(.rect)  // the menu anywhere along the line, not only on its text
        .contextMenu {
            Button("Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(row.copied, forType: .string)
            }
        }
    }

    static var header: some View {
        HStack(spacing: Metrics.Spacing.small) {
            Text("Time").frame(width: Metrics.logTime, alignment: .leading)
            Text("Source").frame(width: Metrics.logSource, alignment: .leading)
            Text("Message").frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.headline)
        .foregroundStyle(.secondary)
        .padding(.horizontal)
        .padding(.vertical, Metrics.Spacing.tight)
    }
}
