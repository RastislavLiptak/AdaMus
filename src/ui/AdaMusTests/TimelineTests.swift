// The timeline: its pyramids draw what the samples hold at any zoom, its scale and ruler, and
// the rows it shares the height of the window between.
import Foundation
import Testing

struct PyramidTests {
    /// Values with no order a pyramid could lean on, the same in every run: the fractions of
    /// the multiples of the golden ratio, spread over -100 to 100.
    private static let samples: [Float] = (0..<5000).map { index in
        let fraction = (Double(index) * 0.618_033_988_749_895).truncatingRemainder(dividingBy: 1)
        return Float(fraction * 200 - 100)
    }

    /// The lowest and highest of the samples from ``from`` up to ``to`` seconds, at 100 a second.
    private func extremes(from: Double, to: Double) -> (Float, Float) {
        let first = max(0, Int((from * 100).rounded()))
        let last = min(Self.samples.count, Int((to * 100).rounded()))
        let part = Self.samples[first..<max(first, last)]
        return (part.min() ?? .infinity, part.max() ?? -.infinity)
    }

    /// A column as wide as a bucket of a level, or the samples themselves: exactly its samples.
    @Test(arguments: [0.01, 0.04, 0.16, 0.64, 2.56])
    func aColumnAsWideAsABucketIsExactlyItsSamples(width: Double) {
        let pyramid = Pyramid(samples: Self.samples, rate: 100, start: 0)
        for column in pyramid.columns(from: 5, to: 45, width: width) where column.time >= 5 && column.time + width <= 45
        {
            let (low, high) = extremes(from: column.time, to: column.time + width)
            #expect((column.low, column.high) == (low, high), "at \(column.time)")
        }
    }

    /// At any other width, a column holds what lies in it, and nothing farther than a bucket
    /// (no wider than the column) beyond it; and the whole span keeps its extremes.
    @Test(arguments: [0.03, 0.25, 1.0, 3.7, 9.9])
    func aColumnOfAnyWidthHoldsItsSamplesAndNoFarther(width: Double) {
        let pyramid = Pyramid(samples: Self.samples, rate: 100, start: 0)
        let columns = pyramid.columns(from: 0, to: 50, width: width)
        for column in columns {
            let (low, high) = extremes(from: column.time - width, to: column.time + 2 * width)
            #expect(column.low >= low && column.high <= high, "at \(column.time)")
        }
        #expect(columns.map(\.low).min() == Self.samples.min())
        #expect(columns.map(\.high).max() == Self.samples.max())
    }

    @Test func aGapIsLeftOutAndAColumnOfSomeNumbersKeepsThem() {
        let samples: [Float] = [1, 2, .nan, .nan, .nan, .nan, 5, .nan]
        let pyramid = Pyramid(samples: samples, rate: 1, start: 0)
        #expect(
            pyramid.columns(from: 0, to: 8, width: 2) == [
                EnvelopePoint(time: 0, low: 1, high: 2), EnvelopePoint(time: 6, low: 5, high: 5),
            ])
    }

    @Test func aSoundTakesItsLowsAndHighsFromItsOwnHalves() {
        let pyramid = Pyramid(low: [-1, -3, -2, -4], high: [1, 2, 3, 2], rate: 1, start: -1)
        // its buckets at -1, 0, 1 and 2 s, in columns of 2 s from -2, 0 and 2 s
        #expect(
            pyramid.columns(from: -1, to: 3, width: 2) == [
                EnvelopePoint(time: -2, low: -1, high: 1), EnvelopePoint(time: 0, low: -3, high: 3),
                EnvelopePoint(time: 2, low: -4, high: 2),
            ])
    }

    @Test func aSpanThatBeginsWithinABucketHasThatBucket() {
        let pyramid = Pyramid(samples: [5, 1, 2, 3], rate: 1, start: 0)
        #expect(pyramid.columns(from: 0.5, to: 2, width: 1).first == EnvelopePoint(time: 0, low: 5, high: 5))
    }

    @Test func eachLevelHasBucketsFourTimesAsLargeAsTheOneBelow() {
        let pyramid = Pyramid(samples: [Float](repeating: 0, count: 1000), rate: 1, start: 0)
        #expect(pyramid.levels.map(\.size) == [4, 16, 64, 256])
        #expect(pyramid.levels.map(\.low.count) == [250, 63, 16, 4])
    }

    @Test func nothingIsDrawnOfNoSpanOrOutsideTheSamples() {
        let pyramid = Pyramid(samples: [1, 2, 3], rate: 1, start: 10)
        #expect(pyramid.columns(from: 5, to: 5, width: 1).isEmpty)
        #expect(pyramid.columns(from: 0, to: 5, width: 1).isEmpty)
        #expect(pyramid.columns(from: 20, to: 30, width: 1).isEmpty)
    }
}

struct TimeScaleTests {
    @Test func timesAndPointsAreTheInverseOfEachOther() {
        let scale = TimeScale(leading: 30, pointsPerSecond: 20)
        #expect(scale.x(of: 31.5) == 30)
        #expect(scale.time(at: 30) == 31.5)
    }

    @Test func theRulerLeavesRoomForItsLabels() {
        #expect(TickSpacing.seconds(forPointsPerSecond: 100) == 1)
        #expect(TickSpacing.seconds(forPointsPerSecond: 10) == 10)
        #expect(TickSpacing.seconds(forPointsPerSecond: 0.001) == 3600)  // the widest there is
    }
}

struct TimelineRowTests {
    private let groups = [
        TrackGroup(
            title: "EEG", colour: .teal,
            tracks: [
                .signal(
                    SignalTrack(id: "a", title: "AF7", caption: nil, scale: 50, rateHz: 256, samples: 1, layers: 1)),
                .signal(
                    SignalTrack(id: "b", title: "AF8", caption: nil, scale: 50, rateHz: 256, samples: 1, layers: 1)),
            ]),
        TrackGroup(
            title: "Focus", colour: .purple,
            tracks: [
                .curve(curve("d", bands: [Band(startS: 0, endS: 1, title: "Rest", colour: .indigo)])),
                .curve(curve("c", bands: [])),
            ]),
    ]

    private static func curve(_ id: String, bands: [Band]) -> CurveTrack {
        CurveTrack(
            id: id, title: id, caption: nil, steps: [], reference: nil, above: nil, below: nil, bands: bands,
            stepped: false)
    }

    @Test func eachRowIsAsHighAsItsKindAndACurveWithAStripHigherWhateverIsFolded() {
        let rows = TimelineRow.rows(of: groups, folded: [])
        let (group, signal, curve) = (Metrics.Row.group, Metrics.Row.signal, Metrics.Row.curve)
        let striped = curve + Metrics.Row.strip
        #expect(rows.map(\.height) == [group, signal, signal, group, striped, curve])
        let folded = TimelineRow.rows(of: groups, folded: ["EEG"])
        #expect(folded.map(\.height) == [group, group, striped, curve])  // the rest no higher than before
    }

    @Test func aFoldedGroupShowsOnlyItsTitle() {
        let rows = TimelineRow.rows(of: groups, folded: ["EEG"])
        #expect(rows.map(\.id) == ["group:EEG", "group:Focus", "track:d", "track:c"])
    }
}

@MainActor
struct TimelineModelTests {
    /// A timeline of ``span`` seconds, its plots 600 points wide.
    private func model(span: Double = 100, tracks: [Track] = []) -> TimelineModel {
        let timeline = Timeline(
            startS: 0, endS: span, groups: [TrackGroup(title: "All", colour: .teal, tracks: tracks)], legend: [],
            consoles: [])
        let model = TimelineModel(timeline: timeline) { URL(string: "http://sound.stub/\($0)")! }
        model.plotWidthChanged(to: 600)
        return model
    }

    @Test func aZoomKeepsTheTimeInTheMiddleOfTheViewWhereItIs() {
        let model = model()
        #expect(model.visibleSeconds == TimelineModel.openingSeconds)
        model.zoom(by: 2)  // 0 to 60 s, then 15 to 45 s
        #expect(model.visibleSeconds == 30 && model.scrollX == 15)
        model.scrollX = 40  // as the operator scrolled: 40 to 70 s
        model.zoom(by: 2)
        #expect(model.visibleSeconds == 15 && model.scrollX == 47.5)
    }

    @Test func aPinchKeepsTheTimeUnderTheFingersWhereItIs() {
        let model = model()
        model.zoom(by: 2, around: 10)  // a sixth of the way across 0 to 60 s, before and after
        #expect(model.visibleSeconds == 30 && model.scrollX == 5)
    }

    @Test func aZoomStopsAtTheNarrowestViewAndAtTheWholeSession() {
        let model = model()
        model.zoom(by: 100)
        #expect(model.visibleSeconds == TimelineModel.narrowestSeconds && !model.canZoomIn && model.canZoomOut)
        model.zoom(by: 0.001)
        #expect(model.visibleSeconds == 100 && model.scrollX == 0 && !model.canZoomOut && model.canZoomIn)
    }

    @Test func aNarrowerOrWiderViewKeepsTheZoomAndWhereItBegins() {
        let model = model()
        model.scrollX = 20
        model.plotWidthChanged(to: 300)  // an inspector slid in: half as much in view
        #expect(model.pointsPerSecond == 10 && model.visibleSeconds == 30 && model.scrollX == 20)
        model.plotWidthChanged(to: 1200)  // wider than the session at this zoom: the whole of it
        #expect(model.visibleSeconds == 100 && model.pointsPerSecond == 12 && model.scrollX == 0)
        model.plotWidthChanged(to: 600)  // the whole session in view stays whole
        #expect(model.visibleSeconds == 100 && model.pointsPerSecond == 6 && !model.canZoomOut)
    }

    @Test func aSlidingSidebarDoesNotTellTheToolbarOfZoomLimitsThatStayed() {
        let model = model()
        nonisolated(unsafe) var told = false
        withObservationTracking {
            _ = model.canZoomIn
            _ = model.canZoomOut
        } onChange: {
            told = true
        }
        for width in stride(from: 590.0, through: 400, by: -10) { model.plotWidthChanged(to: width) }
        #expect(!told)
        model.zoom(by: 0.001)  // the whole session: it can zoom out no more
        #expect(told && !model.canZoomOut)
    }

    @Test func aDragWithShiftMarksThePartItCrossedEitherWayAndItIsAskedAbout() {
        let model = model()  // 10 points a second
        model.mark(fromLanesX: 300, toLanesX: 100)
        #expect(model.marked == MarkedPart(startS: 10, endS: 30) && model.asked == nil)  // while it is dragged
        model.endMarking()
        #expect(model.asked == MarkedPart(startS: 10, endS: 30) && model.marked == model.asked)
        model.forgetMarked()  // its sheet closed
        #expect(model.marked == nil && model.asked == nil)
    }

    @Test func aPartIsMarkedAsFarAsTheSessionReaches() {
        let model = model()
        model.mark(fromLanesX: -50, toLanesX: 5000)
        #expect(model.marked == MarkedPart(startS: 0, endS: 100))
    }

    @Test func aClickWithShiftMarksNothing() {
        let model = model()
        model.mark(fromLanesX: 100, toLanesX: 100 + Double(Metrics.leastMarked) - 1)
        model.endMarking()
        #expect(model.marked == nil && model.asked == nil)
        model.mark(fromLanesX: 100, toLanesX: 100 + Double(Metrics.leastMarked))
        model.endMarking()
        #expect(model.asked == MarkedPart(startS: 10, endS: 10 + Double(Metrics.leastMarked) / 10))
    }

    @Test func aPartIsSaidFromWhenToWhenAsTheTimelineWritesTimesAndHowLong() {
        let said = MarkedPartSheet.span(of: MarkedPart(startS: 62.5, endS: 80))
        let length = 17.5.formatted(.number.precision(.fractionLength(1)))
        #expect(
            said
                == "\(ValueFormatting.clock(62.5, tenths: true)) – \(ValueFormatting.clock(80, tenths: true)) (\(length) s)"
        )
        #expect(said.hasPrefix("1:02") && said.hasSuffix(" s)"))
    }

    @Test func theViewNeverScrollsPastEitherEndOfTheSession() {
        let model = model()
        model.scroll(to: -10)
        #expect(model.scrollX == 0)
        model.scroll(to: 90)
        #expect(model.scrollX == 100 - TimelineModel.openingSeconds)
    }

    @Test func thePlayheadBeginsWhereTheMusicDoesAndAClickOnTheLanesMovesItThere() {
        let audio = AudioTrack(
            id: "sound", title: "Sound", caption: nil, startS: 3, durationS: 50, rateHz: 1, buckets: 50, level: [])
        let model = model(tracks: [.audio(audio)])
        #expect(model.playhead == 3)  // shown before Play, where Play starts
        model.seek(toLanesX: 200)  // 600 points for 60 s: 20 s from the start
        #expect(model.playhead == 20)
    }

    @Test func thePlayheadStaysWithinTheSessionAndIsBroughtIntoView() {
        let model = model()
        model.scrollX = 20
        model.seek(to: -5)
        #expect(model.playhead == 0 && model.scrollX == 0)
        model.seek(to: 150)
        #expect(model.playhead == 100 && model.scrollX == 100 - TimelineModel.openingSeconds)
        model.scrollX = 30
        model.seek(to: 70)  // in view: the view stays
        #expect(model.playhead == 70 && model.scrollX == 30)
    }

    @Test func eachLayerOfASignalAndTheSoundGetPyramidsOfTheirOwnSamples() async throws {
        let signal = SignalTrack(id: "eeg", title: "EEG", caption: nil, scale: 1, rateHz: 1, samples: 3, layers: 2)
        let audio = AudioTrack(
            id: "sound", title: "Sound", caption: nil, startS: 2, durationS: 2, rateHz: 1, buckets: 2, level: [])
        let model = model(tracks: [.signal(signal), .audio(audio)])
        let samples: [String: [Float]] = ["eeg": [1, 2, 3, 10, 20, 30], "sound": [-1, -2, 1, 2]]
        try await model.load { samples[$0] ?? [] }
        #expect(model.pyramids["eeg"]?.map(\.samples.low) == [[1, 2, 3], [10, 20, 30]])
        let sound = try #require(model.pyramids["sound"]?.first)
        #expect(sound.samples.low == [-1, -2] && sound.samples.high == [1, 2] && sound.start == 2)
        #expect(model.soundURL == URL(string: "http://sound.stub/sound"))
    }
}

@MainActor
struct PlaybackTests {
    @Test func aSoundThatCannotBePlayedSaysWhyAndDoesNotPlay() async throws {
        let audio = AudioTrack(
            id: "sound", title: "Sound", caption: nil, startS: 0, durationS: 2, rateHz: 1, buckets: 2, level: [])
        let timeline = Timeline(
            startS: 0, endS: 2, groups: [TrackGroup(title: "Music", colour: .pink, tracks: [.audio(audio)])],
            legend: [], consoles: [])
        let notSound = try SmallProgram.write("not a sound", to: try SmallProgram.newFolder().appending(path: "x.wav"))
        let model = TimelineModel(timeline: timeline) { _ in notSound }
        defer { model.stop() }
        model.togglePlayback()
        try await StubProgram.until("the player has found it cannot play") { model.playbackProblem != nil }
        #expect(!model.playing)
        model.togglePlayback()
        #expect(!model.playing)
    }

    /// The music of a session of 200 s, from its start, its plots 600 points wide: 60 s in view,
    /// 10 points a second. Its sound is never there: the player finds so only once the test waits.
    private func playable() -> TimelineModel {
        let audio = AudioTrack(
            id: "sound", title: "Sound", caption: nil, startS: 0, durationS: 200, rateHz: 1, buckets: 200, level: [])
        let timeline = Timeline(
            startS: 0, endS: 200, groups: [TrackGroup(title: "Music", colour: .pink, tracks: [.audio(audio)])],
            legend: [], consoles: [])
        let model = TimelineModel(timeline: timeline) { _ in URL(filePath: "/nowhere/sound.wav") }
        model.plotWidthChanged(to: 600)
        return model
    }

    @Test func theViewFollowsThePlayheadItShowedAtPlayUntilItIsScrolledOutOfView() {
        let model = playable()
        defer { model.stop() }
        model.togglePlayback()
        #expect(model.playing && model.following)
        model.played(to: 61)  // past the edge: the next page
        #expect(model.scrollX == 61)
        model.scrolled(toOffset: 610.5)  // as the scroll view rounds where it was told to go
        #expect(model.following)
        model.scrolled(toOffset: 550)  // by the operator, the playhead still in view
        #expect(model.following && model.scrollX == 55)
        model.played(to: 70)
        model.scrolled(toOffset: 0)  // the playhead out of view: the view no longer turns back to it
        #expect(!model.following)
        model.played(to: 80)
        #expect(model.scrollX == 0 && model.playhead == 80)
    }

    @Test func aPlayheadOutOfViewAtPlayIsNotFollowedEvenOnceItComesIntoView() {
        let model = playable()
        defer { model.stop() }
        model.seek(to: 80)
        model.scrolled(toOffset: 0)  // 0 to 60 s
        model.togglePlayback()
        #expect(model.playing && !model.following)
        model.played(to: 30)
        model.played(to: 61)
        #expect(model.scrollX == 0)
        model.togglePlayback()  // paused, scrolled to see it, and played again: followed
        model.scrolled(toOffset: 300)
        model.togglePlayback()
        #expect(model.playing && model.following)
    }
}

struct PartitioningTests {
    @Test func theFirstThatBelongsInACollectionOfAnySize() {
        for count in 0...9 {
            let times = Array(0..<count)
            for first in 0...count {
                #expect(times.partitioningIndex { $0 >= first } == first, "\(first) of \(count)")
            }
        }
        let later = [1, 3, 5, 7][1...]  // a slice keeps the indices of what it was cut from
        #expect(later.partitioningIndex { $0 >= 5 } == 2)
    }
}

struct LaneDrawingTests {
    @Test func aDrawingTakesOneStepMoreOnEachSideForTheLineBetweenTheMiddlesToCrossItsEdges() {
        let steps = (0..<10).map { CurveStep(startS: Double($0), endS: Double($0 + 1), value: 0.5, colour: nil) }
        let curve = CurveTrack(
            id: "c", title: "Chance", caption: nil, steps: steps, reference: nil, above: nil, below: nil, bands: [],
            stepped: false)
        #expect(LaneDrawing.steps(of: curve.steps, from: 4.2, to: 6.8).map(\.startS) == [3, 4, 5, 6, 7])
        // none before the first, none after the last
        #expect(LaneDrawing.steps(of: curve.steps, from: 0, to: 1.5).map(\.startS) == [0, 1, 2])
        #expect(LaneDrawing.steps(of: curve.steps, from: 9.5, to: 12).map(\.startS) == [8, 9])
    }

    @Test func theLineBreaksWhereAStepHasNoValueAndItsAreaReachesTheReference() {
        let drawing = LaneDrawing(
            scale: TimeScale(leading: 0, pointsPerSecond: 10), from: 0, to: 4, pyramids: TrackPyramids())
        let steps = [0.75, nil, 0.25, 0.75].enumerated().map {
            CurveStep(startS: Double($0.offset), endS: Double($0.offset + 1), value: $0.element, colour: nil)
        }
        let plot = CGRect(x: 0, y: 0, width: 40, height: 100)
        let parts = drawing.parts(of: steps[...], stepped: false, in: plot)  // through the middles
        #expect(parts == [[CGPoint(x: 5, y: 25)], [CGPoint(x: 25, y: 75), CGPoint(x: 35, y: 25)]])
        let held = drawing.parts(of: steps[...], stepped: true, in: plot)
        #expect(held.first == [CGPoint(x: 0, y: 25), CGPoint(x: 10, y: 25)])
        let area = LaneDrawing.area(of: held, to: 50)
        #expect(area.contains(CGPoint(x: 5, y: 40)))  // between the line and the reference
        #expect(!area.contains(CGPoint(x: 5, y: 20)) && !area.contains(CGPoint(x: 5, y: 60)))  // beyond either
        #expect(!area.contains(CGPoint(x: 15, y: 40)))  // no value: no area
        #expect(area.contains(CGPoint(x: 25, y: 60)))  // under the reference too
    }

    @Test func howFarTheSoundWasSteeredRisesFromTheBottomOfItsLaneAsHighAsEachValue() {
        let drawing = LaneDrawing(
            scale: TimeScale(leading: 0, pointsPerSecond: 10), from: 0, to: 4, pyramids: TrackPyramids())
        let lane = CGRect(x: 0, y: 100, width: 40, height: 81)  // 80 above the hairline
        let steps: [CurveStep] = [
            CurveStep(startS: 0, endS: 1, value: 0.25, colour: nil),
            CurveStep(startS: 1, endS: 2, value: nil, colour: nil),
            CurveStep(startS: 2, endS: 4, value: 1, colour: nil),
        ]
        let area = drawing.level(steps[...], in: lane)
        #expect(area.contains(CGPoint(x: 5, y: 175)) && !area.contains(CGPoint(x: 5, y: 155)))  // a quarter: 20 of 80
        #expect(!area.contains(CGPoint(x: 15, y: 175)))  // none: nothing
        #expect(area.contains(CGPoint(x: 35, y: 101)))  // the whole: up to the top
    }
}

@MainActor
struct TimelineSamplesTests {
    @Test func samplesThatDoNotFitTheirTrackSayWhichTrack() async {
        let signal = SignalTrack(id: "eeg", title: "AF7", caption: nil, scale: 1, rateHz: 1, samples: 3, layers: 1)
        let timeline = Timeline(
            startS: 0, endS: 3, groups: [TrackGroup(title: "EEG", colour: .teal, tracks: [.signal(signal)])],
            legend: [],
            consoles: [])
        let model = TimelineModel(timeline: timeline) { _ in URL(string: "http://sound.stub")! }
        await #expect(throws: TimelineModel.UnfitSamples(track: "AF7")) { try await model.load { _ in [1, 2] } }
    }
}
