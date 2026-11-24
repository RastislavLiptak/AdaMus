// How the app writes values and reads what the program prints: numbers in the person's locale,
// durations in the unit that suits them, the messages of the program line by line.
import Foundation
import Testing

struct ValueFormattingTests {
    private let czech = Locale(identifier: "cs_CZ")
    private let english = Locale(identifier: "en_US")

    @Test func aClockOfMinutesThenOfHours() {
        #expect(ValueFormatting.clock(0) == "0:00")
        #expect(ValueFormatting.clock(65.4) == "1:05")
        #expect(ValueFormatting.clock(3725) == "1:02:05")
        #expect(ValueFormatting.clock(59.6) == "1:00")  // rounded to the whole second shown
    }

    @Test func tenthsAfterTheDecimalSeparatorOfTheLocale() {
        #expect(ValueFormatting.clock(65.36, tenths: true, locale: czech) == "1:05,4")
        #expect(ValueFormatting.clock(65.36, tenths: true, locale: english) == "1:05.4")
        #expect(ValueFormatting.clock(3599.96, tenths: true, locale: english) == "1:00:00.0")
    }

    @Test func aTimeBeforeTheStartHasAMinusSignButZeroHasNone() {
        #expect(ValueFormatting.clock(-2.5, tenths: true, locale: english) == "−0:02.5")
        #expect(ValueFormatting.clock(-0.01, tenths: true, locale: english) == "0:00.0")
    }

    @Test func aDurationInTheUnitThatSuitsItsSize() {
        #expect(ValueFormatting.duration(0.006, locale: czech) == "6 ms")
        #expect(ValueFormatting.duration(1.5, locale: czech) == "1,5\u{a0}s")  // unbroken, as Czech writes it
        #expect(ValueFormatting.duration(85, locale: czech) == "1:25")
    }

    @Test func aDurationIsInTheUnitOfItsSizeAsWrittenNot60SecondsOr1000Milliseconds() {
        #expect(ValueFormatting.duration(59.96, locale: czech) == "1:00")
        #expect(ValueFormatting.duration(0.9996, locale: czech) == "1,0\u{a0}s")
        #expect(ValueFormatting.duration(.nan, locale: czech) == ValueFormatting.unknown)
        #expect(ValueFormatting.clock(.infinity) == ValueFormatting.unknown)
    }

    @Test func aDurationBeforeTheStartInTheUnitOfItsSize() {
        #expect(ValueFormatting.duration(-1.5, locale: czech).hasSuffix("1,5\u{a0}s"))
        #expect(ValueFormatting.duration(-90, locale: czech) == "−1:30")
    }

    @Test func aMomentWithItsDayAndTimeInTheTimeZoneGiven() {
        let moment = Value(.number(1_700_000_000), format: .datetime)  // 22:13:20 UTC on 14 November 2023
        let utc = TimeZone(identifier: "UTC")!
        #expect(ValueFormatting.text(moment, locale: czech, timeZone: utc) == "14.\u{202f}11.\u{202f}2023 22:13")
        #expect(ValueFormatting.text(moment, locale: english, timeZone: utc) == "Nov 14, 2023 at 10:13\u{202f}PM")
        let prague = TimeZone(identifier: "Europe/Prague")!
        #expect(ValueFormatting.text(moment, locale: czech, timeZone: prague).hasSuffix("23:13"))
    }

    @Test func aValueAsItsFormatSays() {
        func text(_ number: Double, _ format: Format) -> String {
            ValueFormatting.text(Value(.number(number), format: format), locale: czech)
        }
        #expect(text(0.333, .percent) == "33\u{a0}%")  // as Czech writes it, the sign unbroken
        #expect(text(153, .count) == "153")
        #expect(text(0.8234, .decimal) == "0,82")
        #expect(text(256, .frequency).hasPrefix("256,0"))
        #expect(ValueFormatting.text(Value(.unknown, format: .count)) == "—")
        #expect(ValueFormatting.text(Value(.text("Muse 2"), format: .text)) == "Muse 2")
    }
}

struct ProgramMessageTests {
    @Test func theChecklistAndHowToFixWhatFailed() {
        let line =
            #"{"checks": [{"name": "Audio output", "status": "failed", "detail": "44100 Hz", "fix": "Set it."}]}"#
        #expect(
            ProgramMessage(line: line)
                == .checks([Check(name: "Audio output", status: .failed, detail: "44100 Hz", fix: "Set it.")]))
    }

    @Test func whereTheApiAnswersAndItsToken() {
        let line = #"{"ready": {"address": "http://127.0.0.1:8000", "token": "secret"}}"#
        #expect(
            ProgramMessage(line: line) == .ready(address: URL(string: "http://127.0.0.1:8000")!, token: "secret"))
    }

    @Test(arguments: ["", "Traceback (most recent call last):", #"{"else": 1}"#, "{"])
    func whatIsNoMessageIsNoMessage(line: String) {
        #expect(ProgramMessage(line: line) == nil)
    }
}

struct LinesTests {
    @Test func eachLineComesAsSoonAsItIsWholeAndTheRestAtTheEnd() async throws {
        let pipe = Pipe()
        var lines = Lines.of(pipe.fileHandleForReading).makeAsyncIterator()
        pipe.fileHandleForWriting.write(Data("first\nsec".utf8))
        #expect(await lines.next() == "first")  // before the writer is done
        pipe.fileHandleForWriting.write(Data("ond\nthird, unended".utf8))
        #expect(await lines.next() == "second")
        try pipe.fileHandleForWriting.close()
        #expect(await lines.next() == "third, unended")
        #expect(await lines.next() == nil)
    }

    @Test func aCarriageReturnEndsALineAsALineFeedDoesAndBothEndOne() async throws {
        let pipe = Pipe()
        let lines = Lines.of(pipe.fileHandleForReading)
        for piece in ["10 %\r20 %\r", "\ndone\r", "\n", "last\r"] { pipe.fileHandleForWriting.write(Data(piece.utf8)) }
        try pipe.fileHandleForWriting.close()
        var read: [String] = []
        for await line in lines { read.append(line) }
        #expect(read == ["10 %", "20 %", "done", "last"])  // \r\n across two pieces is one end
    }

    @Test func aLineInManyPiecesAndManyLinesInOne() async throws {
        let pipe = Pipe()
        let lines = Lines.of(pipe.fileHandleForReading)
        for piece in ["a", "b", "\n", "č\nd\n\ne"] { pipe.fileHandleForWriting.write(Data(piece.utf8)) }
        try pipe.fileHandleForWriting.close()
        var read: [String] = []
        for await line in lines { read.append(line) }
        #expect(read == ["ab", "č", "d", "", "e"])
    }
}

struct ConsoleLineTests {
    @Test func aCopiedLineIsItsTimeItsSourceAndItsTextBetweenTabs() {
        let time = ValueFormatting.clock(65.3, tenths: true)
        let line = ConsoleLine(timeS: 65.3, source: "Detector", severity: .warning, text: "No signal.")
        #expect(ConsolesInspector.Line(id: 0, line: line).copied == "\(time)\tDetector\tNo signal.")
        let unsourced = ConsoleLine(timeS: 65.3, source: nil, severity: .info, text: "Started.")
        #expect(ConsolesInspector.Line(id: 1, line: unsourced).copied == "\(time)\tStarted.")
    }

    @Test func aLogShowsTheLinesThatCameByThePlayheadOneAtItToo() {
        let lines = [0.0, 1, 2, 3].map { ConsoleLine(timeS: $0, source: nil, severity: .info, text: "at \($0)") }
        #expect(ConsolesInspector.heard(lines, by: -1) == 0)  // before the session
        #expect(ConsolesInspector.heard(lines, by: 1.5) == 2)
        #expect(ConsolesInspector.heard(lines, by: 2) == 3)
        #expect(ConsolesInspector.heard(lines, by: 10) == 4)
    }

    @Test func aLogFollowsItsNewestLinesOnlyWhileItIsScrolledToItsEnd() {
        #expect(LogTable.isAtEnd(offset: 400, shown: 200, of: 600))  // at the end
        #expect(LogTable.isAtEnd(offset: 399.5, shown: 200, of: 600))  // less than a point from it
        #expect(!LogTable.isAtEnd(offset: 398, shown: 200, of: 600))  // scrolled up, if only a little
        #expect(LogTable.isAtEnd(offset: 0, shown: 200, of: 50))  // all of it in view
        // only a person scrolling changes whether it follows
        #expect(!LogTable.follows(true, from: .idle, to: .interacting, atEnd: true))  // the log is theirs
        #expect(!LogTable.follows(true, from: .interacting, to: .idle, atEnd: false))  // let go above the end
        #expect(LogTable.follows(false, from: .decelerating, to: .idle, atEnd: true))  // let go at the end
        #expect(LogTable.follows(true, from: .idle, to: .idle, atEnd: false))  // new lines below: still
        #expect(!LogTable.follows(false, from: .idle, to: .idle, atEnd: true))
    }
}
