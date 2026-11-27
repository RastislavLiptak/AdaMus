// A sound played from where it is put on the output the system plays on, and paused or stopped;
// where it is, and back at its start once it has played to its end. Apart from it, a stretch of
// a sound played over and over from its start, as a part being marked is heard: by a player of
// its own, so that the sound itself stops and stays where it was. The library plays the song it
// shows so, wherever it is shown, and the overview the song of a session. One sound at a time:
// what shows another sound shows it at its start, not playing.
import AVFoundation
import Observation

@MainActor @Observable
final class SoundPlayer {
    private(set) var playing = false
    /// Seconds into the sound where it was put, paused, or played to its end.
    private var put = 0.0
    private var player: AVPlayer?
    /// The sound loaded: the one that plays, is paused or was put somewhere.
    private var url: URL?
    /// The seeks asked for, and the last that has landed: until it has, the player still tells
    /// where it was before.
    private var seeks = 0
    private var landed = 0
    private var ended: (any NSObjectProtocol)?
    private var failed: NSKeyValueObservation?
    /// The player of the stretch played over and over, the sound it plays, and the stretch,
    /// while one plays.
    private var looper: AVPlayer?
    private var loopedSound: URL?
    private var looped: ClosedRange<Double>?
    private var loopEnded: (any NSObjectProtocol)?
    /// Looks where the stretch is as it plays, to play it again from its start at its end.
    private var loopWatch: Any?
    private static let loopWatchEvery = CMTime(value: 1, timescale: 50)

    /// Seconds into the sound, now: what the player tells, as it plays, which only what shows
    /// it as it plays asks for (a TimelineView); where it was put, paused or ended otherwise.
    var time: Double {
        guard playing, landed == seeks, let player else { return put }
        return player.currentTime().seconds
    }

    /// Whether the sound at ``url`` plays now.
    func plays(_ url: URL) -> Bool {
        playing && url == self.url
    }

    /// Seconds into the sound at ``url``: where it is, if it is the one loaded; its start if not.
    func time(of url: URL) -> Double {
        url == self.url ? time : 0
    }

    /// Whether a stretch of the sound at ``url`` plays over and over now.
    func loops(_ url: URL) -> Bool {
        looped != nil && url == loopedSound
    }

    /// Seconds into the sound at ``url`` where the stretch played over and over is, while one
    /// plays: within the stretch, also while it goes back to its start.
    func loopTime(of url: URL) -> Double? {
        guard let looped, let looper, url == loopedSound else { return nil }
        return min(max(looper.currentTime().seconds, looped.lowerBound), looped.upperBound)
    }

    /// Play the sound at ``url`` on from where it is put: where it was paused or a seek put it,
    /// at its beginning once it was stopped or played to its end, or another was loaded.
    func play(_ url: URL) {
        endLoop()
        if url != self.url { load(url) }
        player?.play()
        playing = true
    }

    func pause() {
        player?.pause()
        put = time
        playing = false
    }

    /// Put the sound at ``url`` at ``seconds``, playing or not; another sound loaded stops.
    func seek(to seconds: Double, in url: URL) {
        if url != self.url { load(url) }
        put = max(0, seconds)
        seeks += 1
        let seek = seeks
        player?.seek(
            to: CMTime(seconds: put, preferredTimescale: 1000), toleranceBefore: .zero, toleranceAfter: .zero
        ) { [weak self] _ in
            Task { @MainActor in self?.landed = max(self?.landed ?? 0, seek) }
        }
    }

    /// Play the stretch ``range`` of the sound at ``url`` over and over from its start, until
    /// ``endLoop``; the sound itself stops where it is. A stretch that changes as it plays plays
    /// on where it is, while that is in it.
    func loop(_ range: ClosedRange<Double>, in url: URL) {
        if playing { pause() }
        if url != loopedSound { loadLoop(url) }
        guard let looper else { return }
        let restart = looped == nil || !range.contains(looper.currentTime().seconds)
        looped = range
        if restart {
            again()
            looper.play()
        }
    }

    /// Stop the stretch played over and over.
    func endLoop() {
        looper?.pause()
        looped = nil
    }

    /// Stop and forget the sound, and the stretch: played again, it starts at its beginning.
    func stop() {
        player?.pause()
        if let ended { NotificationCenter.default.removeObserver(ended) }
        ended = nil
        failed = nil
        player = nil
        url = nil
        playing = false
        put = 0
        landed = seeks
        forgetLoop()
    }

    private func load(_ url: URL) {
        stop()
        let player = AVPlayer(url: url)
        ended = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: player.currentItem, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.pause()
                self?.seek(to: 0, in: url)
            }
        }
        failed = player.currentItem?.observe(\.status) { [weak self, weak player] item, _ in
            guard item.status == .failed else { return }
            Task { @MainActor in
                if let self, self.player === player { self.stop() }  // not a newer sound
            }
        }
        self.player = player
        self.url = url
    }

    /// A player of its own for the stretches of the sound at ``url``, beside that of the sound,
    /// which stays where it is.
    private func loadLoop(_ url: URL) {
        forgetLoop()
        let looper = AVPlayer(url: url)
        loopEnded = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: looper.currentItem, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.looped != nil else { return }
                self.again()
                self.looper?.play()
            }
        }
        loopWatch = looper.addPeriodicTimeObserver(forInterval: Self.loopWatchEvery, queue: .main) {
            [weak self] at in
            MainActor.assumeIsolated {
                guard let self, let looped = self.looped, at.seconds >= looped.upperBound else { return }
                self.again()
            }
        }
        self.looper = looper
        loopedSound = url
    }

    private func forgetLoop() {
        endLoop()
        if let loopWatch { looper?.removeTimeObserver(loopWatch) }
        if let loopEnded { NotificationCenter.default.removeObserver(loopEnded) }
        loopWatch = nil
        loopEnded = nil
        looper = nil
        loopedSound = nil
    }

    /// The stretch played again from its start.
    private func again() {
        guard let looper, let looped else { return }
        looper.seek(
            to: CMTime(seconds: looped.lowerBound, preferredTimescale: 1000), toleranceBefore: .zero,
            toleranceAfter: .zero)
    }
}
