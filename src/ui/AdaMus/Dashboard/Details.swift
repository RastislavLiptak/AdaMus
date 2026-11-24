// The details of a session: every fact the program knows of it beyond its overview, section
// after section in the order the program gives them, as System Settings shows its settings: a
// grouped form, each fact its name on the left and its value on the right, a table across.
import SwiftUI

struct DetailsView: View {
    let dashboard: Dashboard

    var body: some View {
        Form {
            if dashboard.problem != nil || dashboard.pending != nil {
                Section { DashboardNotes(dashboard: dashboard) }
            }
            ForEach(dashboard.sections, id: \.self) { section in
                Section(section.title) {
                    ForEach(Array(section.items.enumerated()), id: \.offset) { _, item in
                        switch item {
                        case .fact(let fact):
                            LabeledContent(fact.label) {
                                Text(ValueFormatting.text(fact.value)).textSelection(.enabled)
                            }
                        case .table(let table):
                            FactGrid(table: table)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}

/// A table of facts: its columns named over its rows. A column of numbers is aligned on the
/// right, as the digits of numbers line up; the last column of text takes the room the others
/// leave, as the name column of a list in Finder does.
struct FactGrid: View {
    let table: FactTable

    var body: some View {
        if table.rows.isEmpty {
            LabeledContent(table.label, value: Self.none)
        } else {
            let columns = Self.columns(of: table)
            Grid(horizontalSpacing: Metrics.Spacing.large, verticalSpacing: Metrics.Spacing.small) {
                GridRow {
                    ForEach(Array(table.columns.enumerated()), id: \.offset) { index, title in
                        Text(title)
                            .foregroundStyle(.secondary)
                            .modifier(GridCell(column: columns[index]))
                    }
                }
                ForEach(Array(table.rows.enumerated()), id: \.offset) { _, row in
                    Divider().gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        ForEach(Array(row.prefix(columns.count).enumerated()), id: \.offset) { index, cell in
                            Text(ValueFormatting.text(cell))
                                .monospacedDigit()
                                .modifier(GridCell(column: columns[index]))
                        }
                    }
                }
            }
            .textSelection(.enabled)
            .padding(.vertical, Metrics.Spacing.small)
        }
    }

    static let none = "None"

    /// How each column of a table is laid out.
    struct Column: Equatable {
        let numbers: Bool
        let wide: Bool
    }

    nonisolated static func columns(of table: FactTable) -> [Column] {
        let numbers = table.columns.indices.map { index in
            isNumbers(table.rows.compactMap { index < $0.count ? $0[index] : nil })
        }
        let wide = numbers.lastIndex(of: false)
        return numbers.indices.map { Column(numbers: numbers[$0], wide: $0 == wide) }
    }

    /// Whether a column holds numbers only, beside cells of unknown value.
    nonisolated static func isNumbers(_ cells: [Value]) -> Bool {
        let known = cells.filter { $0.content != .unknown }
        return !known.isEmpty && known.allSatisfy { if case .number = $0.content { true } else { false } }
    }
}

/// A cell of a table of facts: as wide as the widest of its column, or as wide as there is room.
struct GridCell: ViewModifier {
    let column: FactGrid.Column

    func body(content: Content) -> some View {
        if column.wide {
            content.frame(maxWidth: .infinity, alignment: .leading)
        } else {
            content
                .fixedSize()
                .gridColumnAlignment(column.numbers ? .trailing : .leading)
        }
    }
}
