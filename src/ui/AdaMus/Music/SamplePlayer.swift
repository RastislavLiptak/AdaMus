// A sample of a style of music, played from where it is put on the output the system plays on,
// and paused or stopped; where it is, and back at its start once it has played to its end. The
// form of a new session and the overview of a session play their samples so.
import AVFoundation
import Observation

@MainActor @Observable
final class SamplePlayer {
    private(set) var playing = false
    /// Seconds into the sample where it was put, paused, or played to its end.
    private var put = 0.0
    private var player: AVPlayer?
    private var url: URL?
    /// The seeks asked for, and the last that has landed: until it has, the player still tells
    /// where it was before.
    private var seeks = 0
    private var landed = 0
    private var ended: (any NSObjectProtocol)?
    private var failed: NSKeyValueObservation?

    /// Seconds into the sample, now: what the player tells, as it plays, which only what shows
    /// it as it plays asks for (a TimelineView); where it was put, paused or ended otherwise.
    var time: Double {
        guard playing, landed == seeks, let player else { return put }
        return player.currentTime().seconds
    }

    /// Play the sample at ``url`` on from where it is put: where it was paused or a seek put it,
    /// at its beginning once it was stopped or played to its end.
    func play(_ url: URL) {
        if url != self.url {
            let from = put  // put there before the sample was loaded
            load(url)
            seek(to: from)
        }
        player?.play()
        playing = true
    }

    func pause() {
        player?.pause()
        put = time
        playing = false
    }

    /// Put the sample at ``seconds``, playing or not.
    func seek(to seconds: Double) {
        put = max(0, seconds)
        seeks += 1
        let seek = seeks
        player?.seek(
            to: CMTime(seconds: put, preferredTimescale: 1000), toleranceBefore: .zero, toleranceAfter: .zero
        ) { [weak self] _ in
            Task { @MainActor in self?.landed = max(self?.landed ?? 0, seek) }
        }
    }

    /// Stop and forget the sample: played again, it starts at its beginning.
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
    }

    private func load(_ url: URL) {
        stop()
        let player = AVPlayer(url: url)
        ended = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: player.currentItem, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.pause()
                self?.seek(to: 0)
            }
        }
        failed = player.currentItem?.observe(\.status) { [weak self, weak player] item, _ in
            guard item.status == .failed else { return }
            Task { @MainActor in
                if let self, self.player === player { self.stop() }  // not a newer sample
            }
        }
        self.player = player
        self.url = url
    }
}
