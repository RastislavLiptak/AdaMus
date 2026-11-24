// What the window does with what the program answers: what it says when the program refuses,
// how it keeps the list of sessions, waits for a dashboard being worked on, offers the form of
// a new session and follows its run. The program is a stand-in here (StubProgram), as these are
// the answers the real one gives only now and then.
import Foundation
import Testing

private func page(_ total: Int, _ ids: [String]) -> String {
    let sessions = ids.map { #"{"id": "\#($0)", "name": "Session \#($0)"}"# }.joined(separator: ", ")
    return #"{"total": \#(total), "sessions": [\#(sessions)]}"#
}

private func dashboard(pending: String?) -> String {
    let pending = pending.map { #""\#($0)""# } ?? "null"
    return #"""
        {"name": "Session a", "problem": null, "pending": \#(pending), "summary": [], "caution": null,
         "highlights": [], "sections": []}
        """#
}

private func runState(_ run: Int, _ phase: String, session: String? = nil, problem: String? = nil) -> String {
    func text(_ value: String?) -> String { value.map { #""\#($0)""# } ?? "null" }
    return #"{"run": \#(run), "phase": "\#(phase)", "session_id": \#(text(session)), "problem": \#(text(problem))}"#
}

struct APIClientTests {
    @Test func whatTheProgramSaysWentWrongIsWhatTheAppSays() async {
        let program = StubProgram()
        program.answer("DELETE api/sessions/a", status: 409, #"{"detail": "The session is still being stored."}"#)
        program.answer(
            "POST api/run", status: 422, #"{"detail": [{"msg": "Field required."}, {"msg": "Too long."}]}"#)
        program.answer("GET api/run", status: 500, "Internal Server Error")
        await #expect(throws: APIError(message: "The session is still being stored.", status: 409)) {
            try await program.api.delete("a")
        }
        await #expect(throws: APIError(message: "Field required. Too long.", status: 422)) {
            try await program.api.start(SessionRequest(sourceId: "s", recording: nil, name: "n"))
        }
        await #expect(throws: APIError(message: "AdaMus did not answer as expected.", status: 500)) {
            try await program.api.runState()
        }
    }

    @Test func everyCallCarriesTheTokenAndASessionIsAskedForInTheWordsOfTheProgram() async throws {
        let program = StubProgram()
        program.answer("POST api/run", runState(1, "starting"))
        _ = try await program.api.start(SessionRequest(sourceId: "dataset", recording: "evening", name: "Morning"))
        let call = try #require(program.made("POST api/run").first)
        #expect(call.token == program.api.token)
        let sent = try JSONSerialization.jsonObject(with: try #require(call.body)) as? [String: String]
        #expect(sent == ["source_id": "dataset", "recording": "evening", "name": "Morning"])
    }
}

@MainActor
struct WorkspaceTests {
    @Test func everyPageIsListedEachSessionOnce() async {
        let program = StubProgram()
        program.answer("GET api/sessions", page(4, ["a", "b"]))
        program.answer("GET api/sessions", page(4, ["b", "c"]))  // a session came meanwhile, and the pages moved
        program.answer("GET api/sessions", page(4, ["d"]))
        let workspace = Workspace(api: program.api)
        await workspace.load()
        #expect(workspace.sessions.map(\.id) == ["a", "b", "c", "d"])
        #expect(program.made("GET api/sessions").map { $0.query["offset"] } == ["0", "2", "3"])
    }

    @Test(.timeLimit(.minutes(1))) func aPageOfNothingNewEndsTheList() async {
        let program = StubProgram()
        program.answer("GET api/sessions", page(3, ["a", "b"]))
        program.answer("GET api/sessions", page(3, []))  // a session went meanwhile
        let workspace = Workspace(api: program.api)
        await workspace.load()
        #expect(workspace.sessions.map(\.id) == ["a", "b"] && workspace.listProblem == nil)
    }

    @Test func aChosenSessionThatIsGoneIsNoLongerChosen() async {
        let program = StubProgram()
        program.answer("GET api/sessions", page(2, ["a", "b"]))
        program.answer("GET api/sessions", page(1, ["a"]))
        let workspace = Workspace(api: program.api)
        await workspace.load()
        workspace.selection = "b"
        #expect(workspace.session?.summary.id == "b")
        await workspace.load()
        #expect(workspace.selection == nil && workspace.session == nil)
    }

    @Test func aDeletedSessionLeavesTheListAndAFailedDeletionSaysWhy() async {
        let program = StubProgram()
        program.answer("GET api/sessions", page(2, ["a", "b"]))
        program.answer("GET api/sessions", page(1, ["b"]))
        program.answer("DELETE api/sessions/a", status: 204, "")
        program.answer("DELETE api/sessions/b", status: 409, #"{"detail": "A session runs."}"#)
        let workspace = Workspace(api: program.api)
        await workspace.load()
        workspace.selection = "a"
        await workspace.delete(workspace.sessions[0])
        #expect(workspace.selection == nil && workspace.sessions.map(\.id) == ["b"] && workspace.problem == nil)

        await workspace.delete(workspace.sessions[0])
        #expect(workspace.problem == "A session runs." && !workspace.isInFront)  // said in an alert
        #expect(workspace.sessions.map(\.id) == ["b"])
    }
}

@MainActor
struct SessionModelTests {
    @Test func aDashboardStillWorkedOnIsReadAgainUntilItIsDoneThenNoMore() async throws {
        let program = StubProgram()
        program.answer("GET api/sessions/a/dashboard", dashboard(pending: "Computing the metrics…"))
        program.answer("GET api/sessions/a/dashboard", dashboard(pending: "Computing the metrics…"))
        program.answer("GET api/sessions/a/dashboard", dashboard(pending: nil))
        let session = SessionModel(api: program.api, summary: SessionSummary(id: "a", name: "Session a"))
        defer { session.stop() }
        try await StubProgram.until("the dashboard is done") {
            session.dashboard != nil && session.dashboard?.pending == nil
        }
        try await Task.sleep(for: APIClient.lookAgain * 2)
        #expect(program.made("GET api/sessions/a/dashboard").count == 3)
        #expect(session.dashboardProblem == nil)
    }

    @Test func aPartIsBuiltOnceItIsFirstShownAndKeptAfter() {
        let session = SessionModel(api: StubProgram().api, summary: SessionSummary(id: "a", name: "Session a"))
        defer { session.stop() }
        #expect(session.opened == [.overview])
        session.part = .timeline
        session.part = .overview
        #expect(session.opened == [.overview, .timeline])
        session.part = .details
        #expect(session.opened == [.overview, .timeline, .details])
    }

    @Test func aDashboardThatCannotBeReadSaysWhy() async throws {
        let program = StubProgram()
        program.answer("GET api/sessions/a/dashboard", status: 404, #"{"detail": "No session a."}"#)
        let session = SessionModel(api: program.api, summary: SessionSummary(id: "a", name: "Session a"))
        defer { session.stop() }
        try await StubProgram.until("the problem is said") { session.dashboardProblem != nil }
        #expect(session.dashboardProblem == "No session a." && session.dashboard == nil)
    }
}

@MainActor
struct NewSessionModelTests {
    private static let form = #"""
        {"sources": [
           {"id": "headset", "title": "Muse 2", "live": true, "recordings": []},
           {"id": "first", "title": "First", "live": false, "recordings": ["f1", "f2"]},
           {"id": "second", "title": "Second", "live": false, "recordings": ["s1", "s2"]}],
         "sources_hint": "No source.", "name_pattern": "[a-z]+", "name_hint": "Small letters only."}
        """#
    private static let fine = #"{"warning": null, "problem": null}"#

    private func opened(refresh: (status: Int, body: String) = (200, form)) async -> (StubProgram, NewSessionModel) {
        let program = StubProgram()
        program.answer("GET api/new-session", Self.form)
        program.answer("POST api/new-session/refresh", status: refresh.status, refresh.body)
        program.answer("GET api/sources/headset/check", Self.fine)  // the first source, chosen at first
        let model = NewSessionModel(api: program.api)
        await model.open()
        return (program, model)
    }

    @Test func anotherSourceStartsFromItsFirstRecordingNotFromTheOneChosenBefore() async {
        let (_, model) = await opened()
        model.chosenSource = "first"
        model.chosenRecording = "f2"
        #expect(model.recording == "f2")
        model.chosenSource = "second"
        #expect(model.recording == "s1")
        model.chosenSource = "first"
        #expect(model.recording == "f1")
    }

    @Test func aSourceTheProgramRefusesCannotStartAndOneItWarnsOfCan() async {
        let (program, model) = await opened()
        model.chosenSource = "first"
        model.name = "morning"
        program.answer("GET api/sources/first/check", #"{"warning": null, "problem": "It has no AF7."}"#)
        program.answer("GET api/sources/first/check", #"{"warning": "It is short.", "problem": null}"#)
        await model.checkChoice()
        #expect(model.check?.problem == "It has no AF7." && !model.canStart)
        await model.checkChoice()
        #expect(model.check?.warning == "It is short." && model.canStart)
        #expect(program.made("GET api/sources/first/check").last?.query == ["recording": "f1"])
    }

    @Test func aFormOpenedWhileTheHeadsetIsLookedForAlreadyStandsWithoutAnAlert() async {
        let (_, model) = await opened(refresh: (409, #"{"detail": "The headset is being looked for."}"#))
        #expect(model.problem == nil && model.form != nil && !model.lookingForHeadset)
        let (_, failed) = await opened(refresh: (500, #"{"detail": "Bluetooth is off."}"#))
        #expect(failed.problem == "Bluetooth is off.")
    }
}

@MainActor
struct RunModelTests {
    @Test func aRunHasEndedOnceItIsIdleOrAnotherHasBeenAskedFor() async {
        let program = StubProgram()
        program.answer("GET api/run", runState(1, "finishing"))
        program.answer("GET api/run", runState(2, "starting"))
        let run = RunModel(api: program.api, state: RunState(run: 1, phase: .running, sessionId: nil, problem: nil))
        await run.refresh()
        #expect(!run.ended)
        await run.refresh()
        #expect(run.ended)
    }

    @Test(arguments: [
        (RunState(run: 1, phase: .idle, sessionId: "s", problem: nil), RunModel.Outcome.stored("s")),
        (RunState(run: 1, phase: .idle, sessionId: "s", problem: "No signal."), .failed("No signal.")),
        (RunState(run: 1, phase: .idle, sessionId: nil, problem: nil), .failed(RunModel.notStored)),
    ])
    func aRunEndsWithItsSessionStoredOrWithWhyNot(state: RunState, outcome: RunModel.Outcome) {
        #expect(RunModel(api: StubProgram().api, state: state).outcome == outcome)
    }

    @Test func aQuestionTheProgramDidNotAnswerIsAskedAgainAndTheRunGoesOn() async {
        let program = StubProgram()
        program.answer("GET api/run", status: 500, #"{"detail": "Busy."}"#)
        program.answer("GET api/run", runState(1, "idle", session: "s"))
        let run = RunModel(api: program.api, state: RunState(run: 1, phase: .running, sessionId: nil, problem: nil))
        await run.refresh()
        #expect(run.problem == "Busy." && !run.ended)  // said, and followed on
        await run.follow()
        #expect(run.outcome == .stored("s") && run.problem == nil)
    }

    @Test func aStopTheProgramRefusedCanBeAskedForAgain() async {
        let program = StubProgram()
        program.answer("POST api/run/stop", status: 500, #"{"detail": "Busy."}"#)
        program.answer("POST api/run/stop", status: 204, "")
        let run = RunModel(api: program.api, state: RunState(run: 1, phase: .running, sessionId: nil, problem: nil))
        await run.stop()
        #expect(!run.stopping && run.problem == "Busy.")
        await run.stop()
        #expect(run.stopping && program.made("POST api/run/stop").count == 2)
    }
}
