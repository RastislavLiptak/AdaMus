// A session that has been asked for, followed until it has ended and been analysed: the program
// is asked how it goes, again and again. A question it does not answer is asked again: the
// session goes on meanwhile, and only the program ending ends it (the window then says so).
import Foundation
import Observation

@MainActor @Observable
final class RunModel {
    let api: APIClient
    let run: Int
    private(set) var state: RunState
    private(set) var stopping = false
    /// What the program did not answer last, while it has not answered since.
    private(set) var problem: String?

    init(api: APIClient, state: RunState) {
        self.api = api
        self.run = state.run
        self.state = state
    }

    /// The run has ended: it is idle again, or another has been asked for since.
    var ended: Bool { state.run != run || state.phase == .idle }

    /// How a run that has ended came out: its session stored, or why there is none.
    enum Outcome: Equatable {
        case stored(String)
        case failed(String)
    }

    nonisolated static let notStored = "The session ended before it was stored."

    var outcome: Outcome {
        if let id = state.sessionId, state.problem == nil { return .stored(id) }
        return .failed(state.problem ?? Self.notStored)
    }

    /// Follow the run until it has ended.
    func follow() async {
        while !ended && !Task.isCancelled {
            try? await Task.sleep(for: APIClient.lookAgain)
            await refresh()
        }
    }

    /// How the run goes now.
    func refresh() async {
        do {
            state = try await api.runState()
            problem = nil
        } catch {
            problem = error.localizedDescription
        }
    }

    /// Stop the session; one the program did not stop can be stopped again.
    func stop() async {
        stopping = true
        do {
            try await api.stopRun()
        } catch {
            stopping = false
            problem = error.localizedDescription
        }
    }
}
