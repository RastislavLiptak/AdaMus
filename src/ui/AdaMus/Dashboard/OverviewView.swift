// The overview of a session: what it is first, when, how long and on what it ran and how it
// ended, the result marked as it went; what bears on how its results are read; then its main
// results, as large as they matter (HIG "Charting data": the macro level first, the details on
// a page of their own); then the electrodes on the head. Lazy: only what is in view is laid out, as a sliding sidebar makes its
// text wrap again in every frame.
import SwiftUI

struct OverviewView: View {
    let dashboard: Dashboard
    let electrodes: ElectrodeSet?

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: Metrics.Spacing.page) {
                DashboardNotes(dashboard: dashboard)
                if !dashboard.summary.isEmpty {
                    Summary(facts: dashboard.summary)
                }
                if let caution = dashboard.caution {  // on yellow, as it is to be seen
                    Label(caution, systemImage: "exclamationmark.triangle.fill")
                        .symbolRenderingMode(.multicolor)
                        .padding(Metrics.Spacing.regular)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.yellow.quaternary, in: .rect(cornerRadius: Metrics.cornerRadius))
                }
                Results(highlights: dashboard.highlights)
                if let electrodes, !electrodes.placed.isEmpty {
                    ElectrodesBox(set: electrodes)
                }
            }
            .padding(Metrics.Spacing.page)
        }
    }
}

/// Why some of a session could not be read, and what is still worked on, over its overview and
/// its details alike.
struct DashboardNotes: View {
    let dashboard: Dashboard

    var body: some View {
        if let problem = dashboard.problem {
            Label(problem, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.secondary)
        }
        if let pending = dashboard.pending {
            Label {
                Text(pending)
            } icon: {
                ProgressView().controlSize(.small)
            }
            .foregroundStyle(.secondary)
        }
    }
}

/// What the session is: each fact its name small over its value, side by side as many as fit; a
/// fact that tells how the session went with its symbol, in its colour.
struct Summary: View {
    let facts: [Fact]

    var body: some View {
        LazyVGrid(
            columns: [
                GridItem(
                    .adaptive(minimum: Metrics.summaryColumn), spacing: Metrics.Spacing.page, alignment: .topLeading)
            ],
            alignment: .leading, spacing: Metrics.Spacing.regular
        ) {
            ForEach(facts, id: \.self) { fact in
                VStack(alignment: .leading, spacing: Metrics.Spacing.line) {
                    Text(fact.label).font(.caption).foregroundStyle(.secondary)
                    let value = ValueFormatting.text(fact.value)
                    if let status = fact.status {
                        Label(value, systemImage: status.systemName).foregroundStyle(status.color)
                    } else {
                        Text(value).textSelection(.enabled)
                    }
                }
                .fontWeight(.medium)
                .accessibilityElement(children: .combine)
            }
        }
    }
}

/// The main results, all in one row where they fit, otherwise in rows as full as each other;
/// each row as high as its highest box.
struct Results: View {
    let highlights: [Highlight]
    @State private var width: CGFloat = 0

    var body: some View {
        let inARow = Self.inARow(count: highlights.count, width: width)
        Grid(horizontalSpacing: Metrics.Spacing.regular, verticalSpacing: Metrics.Spacing.regular) {
            ForEach(Array(stride(from: 0, to: highlights.count, by: inARow)), id: \.self) { first in
                GridRow(alignment: .top) {
                    ForEach(highlights[first..<min(first + inARow, highlights.count)], id: \.self) {
                        ResultBox(highlight: $0)
                    }
                }
            }
        }
        .onGeometryChange(for: CGFloat.self) {
            $0.size.width
        } action: {
            width = $0
        }
    }

    /// How many boxes go in a row: as many as fit, no narrower than a box may be, shared so
    /// that the rows are as full as each other (five that fit four to a row: three and two).
    nonisolated static func inARow(count: Int, width: CGFloat) -> Int {
        let spacing = Metrics.Spacing.regular
        let fit = max(1, Int((width + spacing) / (Metrics.resultBox + spacing)))
        let rows = max(1, (count + fit - 1) / fit)
        return max(1, (count + rows - 1) / rows)
    }
}

/// A main result in a box of its own: what it is, by a dot of the colour of its kind (as the
/// groups of the timeline are marked), its value large in that colour, and what the value means.
/// The colour is never all that tells it: the title names it (HIG "Color").
struct ResultBox: View {
    let highlight: Highlight

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Metrics.Spacing.tight) {
                Label {
                    Text(highlight.title)
                } icon: {
                    Image(systemName: "circle.fill").imageScale(.small).foregroundStyle(highlight.colour.color)
                }
                .font(.headline)
                Text(ValueFormatting.text(highlight.value))
                    .font(.largeTitle)
                    .fontWeight(.semibold)
                    .monospacedDigit()
                    .foregroundStyle(highlight.colour.color)
                Text(highlight.caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(Metrics.Spacing.small)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .accessibilityElement(children: .combine)
    }
}
