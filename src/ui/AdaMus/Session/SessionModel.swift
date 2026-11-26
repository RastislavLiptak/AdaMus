// One session, as the window shows it: its dashboard, of its overview and its details (read
// again while the program still works on it), and its timeline, kept while the session is chosen, so that switching between them
// loses nothing, not even the music playing.
import Foundation
import Observation

/// What the detail of a session shows.
enum SessionPart: String, CaseIterable, Identifiable {
    case overview, timeline, details

    var id: Self { self }

    var title: String {
        switch self {
        case .overview: "Overview"
        case .timeline: "Timeline"
        case .details: "Details"
        }
    }
}

@MainActor @Observable
final class SessionModel {
    let api: APIClient
    let summary: SessionSummary
    /// What is shown; as one part is left for another, what it played is paused.
    var part: SessionPart = .overview {
        didSet {
            opened.insert(part)
            if part != oldValue { pause() }
        }
    }
    /// The sample of the session's music, played on its overview.
    let sample = SamplePlayer()
    /// The parts shown so far: each stays built, so that switching back to one is at once.
    private(set) var opened: Set<SessionPart> = [.overview]
    private(set) var dashboard: Dashboard?
    private(set) var dashboardProblem: String?
    private(set) var timeline: TimelineModel?
    private(set) var timelineProblem: String?
    /// The electrodes of the session on the head, once read; none where they cannot be.
    private(set) var electrodes: ElectrodeSet?
    private var loading: [Task<Void, Never>] = []

    init(api: APIClient, summary: SessionSummary) {
        self.api = api
        self.summary = summary
        loading = [
            Task { await self.readDashboard() }, Task { await self.readTimeline() },
            Task { await self.readElectrodes() },
        ]
    }

    /// Nothing of the session plays: its timeline and the sample of its music are paused.
    func pause() {
        timeline?.pause()
        if sample.playing { sample.pause() }
    }

    /// The session is no longer chosen: nothing more is read, and its music stops.
    func stop() {
        for task in loading { task.cancel() }
        timeline?.stop()
        sample.stop()
    }

    /// The dashboard, read again while the program is still working on it.
    private func readDashboard() async {
        while !Task.isCancelled {
            do {
                let read = try await api.dashboard(of: summary.id)
                dashboard = read
                guard read.pending != nil else { return }
            } catch {
                if !Task.isCancelled { dashboardProblem = error.localizedDescription }
                return
            }
            try? await Task.sleep(for: APIClient.lookAgain)
        }
    }

    /// The electrodes and the head; a session whose electrodes cannot be read shows none, its
    /// dashboard says why.
    private func readElectrodes() async {
        async let head = api.head()
        async let electrodes = api.electrodes(of: summary.id)
        if let head = try? await head, let electrodes = try? await electrodes {
            self.electrodes = ElectrodeSet(head: head, electrodes: electrodes)
        }
    }

    private func readTimeline() async {
        do {
            let read = try await api.timeline(of: summary.id)
            let id = summary.id
            let model = TimelineModel(timeline: read) { [api] track in api.soundURL(of: track, in: id) }
            timeline = model
            try await model.load { [api] track in try await api.samples(of: track, in: id) }
        } catch {
            if !Task.isCancelled { timelineProblem = error.localizedDescription }
        }
    }
}
