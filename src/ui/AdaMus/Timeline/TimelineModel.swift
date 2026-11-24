// What the timeline shows and where: the samples of its tracks, read once and kept as pyramids
// to draw at any zoom; the span of time in view, scrolled and zoomed; and the playhead, which
// follows the music while it plays.
import AVFoundation
import Observation
import SwiftUI

/// The times across the plots: the time at their leading edge, and how many points a second
/// takes. A drawing reads it once, not for each point.
struct TimeScale: Equatable, Sendable {
    let leading: Double
    let pointsPerSecond: Double

    func x(of time: Double) -> Double { (time - leading) * pointsPerSecond }
    func time(at x: Double) -> Double { leading + x / max(pointsPerSecond, .leastNonzeroMagnitude) }
}

/// The distance between two labelled times: the shortest of the usual ones that leaves room.
enum TickSpacing {
    static let steps: [Double] = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
    static let leastPoints = Double(Metrics.tickSpacing)

    static func seconds(forPointsPerSecond pps: Double) -> Double {
        steps.first { $0 * pps >= leastPoints } ?? steps[steps.count - 1]
    }

}

/// A part of the session, marked to see its data: from one time to a later one.
struct MarkedPart: Identifiable, Equatable {
    let startS: Double
    let endS: Double

    var id: [Double] { [startS, endS] }
}

@MainActor @Observable
final class TimelineModel {
    let timeline: Timeline
    let soundURL: URL?
    private let audio: AudioTrack?

    /// The pyramid of every layer of every signal and of the sound, by the track.
    private(set) var pyramids = TrackPyramids()
    /// Seconds of the session across the width of the plots.
    private(set) var visibleSeconds: Double {
        didSet { zoomLimitsChanged() }
    }
    /// Whether the view can zoom in or out, kept apart from the seconds: the toolbar reads it, and
    /// was built again in every frame of a sliding sidebar while it read the seconds themselves.
    private(set) var canZoomIn = false
    private(set) var canZoomOut = false
    /// The width of the plots, in points.
    private(set) var plotWidth: Double = 0
    /// The time at the leading edge of the plots, as the scroll view says.
    var scrollX: Double
    /// Where the scroll view of the lanes is scrolled to, in points: set together with the zoom,
    /// so that the lanes are never drawn at the new zoom from the old place.
    var position = ScrollPosition(edge: .leading)
    /// Where the playhead is, in seconds of the session.
    private(set) var playhead: Double
    private(set) var playing = false
    /// Whether the view turns its pages after the playhead while it plays: so if it showed the
    /// playhead when Play was pressed, until the operator scrolls it out of view.
    private(set) var following = false
    /// Why the sound cannot be played, once the player has found it cannot.
    private(set) var playbackProblem: String?

    /// The part the operator marks with Shift and a drag, while it is marked and asked about.
    private(set) var marked: MarkedPart?
    /// The part marked, once the drag has ended: the sheet of its data is open.
    var asked: MarkedPart?

    private var player: AVPlayer?
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?
    private var statusObserver: NSKeyValueObservation?

    static let zoomStep = 2.0
    static let narrowestSeconds = 5.0
    static let openingSeconds = 60.0  // in view when the timeline opens, at most
    static let playheadLook = CMTime(value: 1, timescale: 60)

    init(timeline: Timeline, soundURL: (String) -> URL) {
        self.timeline = timeline
        let audio = timeline.groups.flatMap(\.tracks).lazy.compactMap { track -> AudioTrack? in
            if case .audio(let audio) = track { audio } else { nil }
        }.first
        self.audio = audio
        self.soundURL = audio.map { soundURL($0.id) }
        visibleSeconds = min(Self.openingSeconds, max(timeline.endS - timeline.startS, Self.narrowestSeconds))
        scrollX = timeline.startS
        playhead = audio?.startS ?? timeline.startS  // where Play starts: the music begins later
        zoomLimitsChanged()
    }

    /// Observation tells those who read a limit only when it changes, not as the seconds do.
    private func zoomLimitsChanged() {
        canZoomIn = visibleSeconds > Self.narrowestSeconds
        canZoomOut = visibleSeconds < span
    }

    var span: Double { max(timeline.endS - timeline.startS, Self.narrowestSeconds) }
    var pointsPerSecond: Double { plotWidth > 0 ? plotWidth / visibleSeconds : 0 }
    var scale: TimeScale { TimeScale(leading: scrollX, pointsPerSecond: pointsPerSecond) }
    var canPlay: Bool { soundURL != nil }

    /// Read the samples of every signal and of the sound, and build their pyramids, away from
    /// the main thread.
    func load(_ read: (String) async throws -> [Float]) async throws {
        var fetched: [String: [Float]] = [:]
        for track in timeline.groups.flatMap(\.tracks) {
            switch track {
            case .signal(let signal): fetched[signal.id] = try await read(signal.id)
            case .audio(let audio): fetched[audio.id] = try await read(audio.id)
            default: break
            }
        }
        let tracks = timeline.groups.flatMap(\.tracks)
        let samples = fetched
        let building = Task.detached(priority: .userInitiated) {
            TrackPyramids(try Self.pyramids(of: tracks, samples: samples))
        }
        // a session no longer chosen builds no more
        pyramids = try await withTaskCancellationHandler {
            try await building.value
        } onCancel: {
            building.cancel()
        }
    }

    /// The plots got wider or narrower, as a sidebar or the inspector slides: the zoom stays and
    /// more or less of the session is in view, from the same time, so that what is drawn is only
    /// moved, not drawn again in every frame of the slide. The whole session in view stays whole,
    /// across the width, and no more than it is ever in view.
    func plotWidthChanged(to width: Double) {
        guard abs(width - plotWidth) >= 1 else { return }
        if plotWidth > 0 {
            visibleSeconds = visibleSeconds >= span ? span : min(span, visibleSeconds * width / plotWidth)
        }
        plotWidth = width
        scroll(to: scrollX)
    }

    /// Zoom in by ``factor`` (out by its inverse), keeping the time at ``focus`` where it is on
    /// the screen: under the fingers of a pinch, or by default the middle of the view.
    func zoom(by factor: Double, around focus: Double? = nil) {
        let anchor = focus ?? scrollX + visibleSeconds / 2
        let share = (anchor - scrollX) / visibleSeconds
        visibleSeconds = min(span, max(Self.narrowestSeconds, visibleSeconds / factor))
        scroll(to: anchor - share * visibleSeconds)
    }

    /// The scroll view was scrolled to ``offset`` points across the lanes: by the operator, or as
    /// it was told. A scroll that leaves the playhead out of view, while the view follows it,
    /// stops the following, so that the view does not turn back to it against the operator.
    func scrolled(toOffset offset: Double) {
        guard pointsPerSecond > 0 else { return }
        scrollX = timeline.startS + offset / pointsPerSecond
        if following, !shows(playhead) { following = false }
    }

    /// Whether ``time`` is in view: give or take the width of the playhead's line, as far as a
    /// scroll view rounds where it is scrolled to.
    func shows(_ time: Double) -> Bool {
        let slack = Double(Metrics.playhead) / max(pointsPerSecond, .leastNonzeroMagnitude)
        return time >= scrollX - slack && time <= scrollX + visibleSeconds + slack
    }

    /// Scroll the view to begin at ``time``, as far as the session reaches.
    func scroll(to time: Double) {
        guard pointsPerSecond > 0 else { return }
        scrollX = max(timeline.startS, min(time, timeline.startS + span - visibleSeconds))
        position.scrollTo(x: (scrollX - timeline.startS) * pointsPerSecond)
    }

    // MARK: Playback

    func togglePlayback() {
        if player == nil, let soundURL { prepare(soundURL) }
        guard let player, let audio, playbackProblem == nil else { return }
        if playing {
            player.pause()
            playing = false
        } else {
            let from = playhead - audio.startS
            player.seek(to: CMTime(seconds: from >= 0 && from < audio.durationS ? from : 0, preferredTimescale: 1000))
            player.play()
            playing = true
            following = shows(playhead)
        }
    }

    func pause() {
        if playing { togglePlayback() }
    }

    /// Seek to the time at ``x`` points across the lanes of the whole session.
    func seek(toLanesX x: Double) {
        guard pointsPerSecond > 0 else { return }
        seek(to: timeline.startS + x / pointsPerSecond)
    }

    func seek(to time: Double) {
        playhead = min(max(timeline.startS, time), timeline.endS)
        if let player, let audio {
            player.seek(to: CMTime(seconds: max(0, playhead - audio.startS), preferredTimescale: 1000))
        }
        if playhead < scrollX || playhead > scrollX + visibleSeconds { scroll(to: playhead - visibleSeconds / 2) }
    }

    // MARK: A part marked

    /// Mark the part between two places across the lanes of the whole session, as a drag with
    /// Shift goes, either way; as far as the session reaches.
    func mark(fromLanesX start: Double, toLanesX end: Double) {
        guard pointsPerSecond > 0 else { return }
        let times = [start, end].sorted().map {
            min(max(timeline.startS, timeline.startS + $0 / pointsPerSecond), timeline.endS)
        }
        marked = MarkedPart(startS: times[0], endS: times[1])
    }

    /// The drag has ended: a part as wide as can be seen is asked about; a click marks nothing.
    func endMarking() {
        if let marked, (marked.endS - marked.startS) * pointsPerSecond >= Metrics.leastMarked {
            asked = marked
        } else {
            marked = nil
        }
    }

    /// The sheet of the part has closed: it is marked no more.
    func forgetMarked() {
        marked = nil
        asked = nil
    }

    /// The timeline is no longer shown: the music stops and is let go.
    func stop() {
        player?.pause()
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        statusObserver?.invalidate()
        timeObserver = nil
        endObserver = nil
        statusObserver = nil
        player = nil
        playing = false
    }

    private func prepare(_ url: URL) {
        let player = AVPlayer(url: url)
        timeObserver = player.addPeriodicTimeObserver(forInterval: Self.playheadLook, queue: .main) {
            [weak self] time in
            guard time.isNumeric else { return }
            MainActor.assumeIsolated { self?.played(to: time.seconds) }
        }
        endObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: player.currentItem, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, let audio = self.audio else { return }
                self.playing = false
                self.playhead = audio.startS + audio.durationS  // at its end, so that Play starts it again
            }
        }
        statusObserver = player.currentItem?.observe(\.status) { [weak self] item, _ in
            guard item.status == .failed else { return }
            let why = item.error?.localizedDescription ?? "The sound could not be played."
            Task { @MainActor in
                self?.playing = false
                self?.playbackProblem = why
            }
        }
        self.player = player
    }

    /// The player has played to ``seconds`` of the sound, as its clock tells.
    func played(to seconds: Double) {
        guard playing, let audio else { return }
        playhead = audio.startS + seconds
        if following, playhead > scrollX + visibleSeconds || playhead < scrollX {
            scroll(to: playhead)  // the next page, as an editor turns it
        }
    }

    // MARK: Pyramids

    /// Samples the program sent that do not make the track it describes.
    struct UnfitSamples: LocalizedError, Equatable {
        let track: String

        var errorDescription: String? { "The samples of \(track) do not fit the track." }
    }

    nonisolated private static func pyramids(of tracks: [Track], samples: [String: [Float]]) throws -> [String:
        [Pyramid]]
    {
        var built: [String: [Pyramid]] = [:]
        for track in tracks {
            try Task.checkCancellation()
            switch track {
            case .signal(let signal):
                guard let values = samples[signal.id], values.count == signal.samples * signal.layers, signal.rateHz > 0
                else { throw UnfitSamples(track: signal.title) }
                built[signal.id] = (0..<signal.layers).map { layer in
                    let slice = values[(layer * signal.samples)..<((layer + 1) * signal.samples)]
                    return Pyramid(samples: Array(slice), rate: signal.rateHz, start: 0)
                }
            case .audio(let audio):
                guard let values = samples[audio.id], values.count == 2 * audio.buckets, audio.rateHz > 0
                else { throw UnfitSamples(track: audio.title) }
                built[audio.id] = [
                    Pyramid(
                        low: Array(values[0..<audio.buckets]), high: Array(values[audio.buckets...]),
                        rate: audio.rateHz, start: audio.startS)
                ]
            case .curve:
                continue
            }
        }
        return built
    }
}
