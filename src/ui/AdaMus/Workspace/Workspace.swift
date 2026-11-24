// What the window works with once the program is ready: the sessions, the one chosen, a new
// session being made, and a deletion being confirmed.
import Foundation
import Observation

@MainActor @Observable
final class Workspace {
    let api: APIClient
    private(set) var sessions: [SessionSummary] = []
    /// Why the list of sessions could not be read, if it could not.
    private(set) var listProblem: String?
    /// The session chosen in the sidebar.
    var selection: SessionSummary.ID? {
        didSet { if selection != oldValue { open() } }
    }
    private(set) var session: SessionModel?
    var isMakingSession = false
    /// Whether the inspector is shown, whichever session is chosen.
    var showsInspector = false
    /// The session whose deletion waits to be confirmed.
    var toDelete: SessionSummary?
    /// What went wrong with something asked for, to be said in an alert.
    var problem: String?

    init(api: APIClient) {
        self.api = api
    }

    /// No sheet or alert lies over the window.
    var isInFront: Bool {
        !isMakingSession && toDelete == nil && problem == nil && session?.timeline?.asked == nil
    }

    /// The sessions again, every page of them; the chosen one stays chosen while it is there.
    func load() async {
        do {
            var page = try await api.sessions(from: 0)
            var listed = page.sessions
            while listed.count < page.total {  // sessions may come or go meanwhile, and pages move
                page = try await api.sessions(from: listed.count)
                let known = Set(listed.map(\.id))
                let new = page.sessions.filter { !known.contains($0.id) }
                if new.isEmpty { break }
                listed += new
            }
            sessions = listed
            listProblem = nil
            if let selection, !sessions.contains(where: { $0.id == selection }) {
                self.selection = nil
            }
        } catch {
            listProblem = error.localizedDescription
        }
    }

    func makeSession() {
        session?.timeline?.pause()  // a session played back would sound over the new one
        isMakingSession = true
    }

    /// A session has ended and been analysed: it heads the list, chosen.
    func sessionEnded(_ id: String) async {
        await load()
        selection = id
    }

    func askToDelete(_ id: SessionSummary.ID?) {
        toDelete = sessions.first { $0.id == id }
    }

    func delete(_ session: SessionSummary) async {
        do {
            try await api.delete(session.id)
            if selection == session.id { selection = nil }
            await load()
        } catch {
            problem = error.localizedDescription
        }
    }

    private func open() {
        session?.stop()
        session = sessions.first { $0.id == selection }.map { SessionModel(api: api, summary: $0) }
    }
}
