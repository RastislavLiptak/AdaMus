// How the overview shares its main results between rows, and how a table of the details aligns
// its columns: numbers on the right, the last column of text taking the room.
import CoreGraphics
import Testing

struct DetailsTests {
    @Test func fiveResultsGoInOneRowWhereTheyFitOtherwiseInRowsAsFullAsEachOther() {
        let box = Metrics.resultBox + Metrics.Spacing.regular
        #expect(Results.inARow(count: 5, width: 5 * box) == 5)
        #expect(Results.inARow(count: 5, width: 4 * box) == 3)  // three and two, not four and one
        #expect(Results.inARow(count: 5, width: 2 * box) == 2)
        #expect(Results.inARow(count: 5, width: 0) == 1)  // before the width is known
        #expect(Results.inARow(count: 0, width: 5 * box) == 1)
    }

    @Test func aColumnOfNumbersIsAlignedOnTheRightWhateverIsUnknownInIt() {
        let number = Value(.number(4), format: .duration)
        let unknown = Value(.unknown, format: .duration)
        let text = Value(.text("Until stopped"), format: .text)
        #expect(FactGrid.isNumbers([number, unknown, number]))
        #expect(!FactGrid.isNumbers([number, text]))
        #expect(!FactGrid.isNumbers([unknown, unknown]))  // nothing to align
    }

    @Test func theLastColumnOfTextTakesTheRoomAndNumbersAreOnTheRight() {
        let number = Value(.number(1), format: .count)
        let text = Value(.text("Rest"), format: .text)
        let table = FactTable(
            label: "Blocks", columns: ["Block", "State", "Action", "Length"],
            rows: [[number, text, text, number], [number, text, text, Value(.unknown, format: .duration)]])
        #expect(
            FactGrid.columns(of: table) == [
                .init(numbers: true, wide: false), .init(numbers: false, wide: false),
                .init(numbers: false, wide: true), .init(numbers: true, wide: false),
            ])
    }
}
