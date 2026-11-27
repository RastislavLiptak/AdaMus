// What the window does with what the program answers: what it says when the program refuses,
// how it keeps the list of sessions, waits for a dashboard being worked on, offers the form of
// a new session and follows its run. The program is a stand-in here (StubProgram), as these are
// the answers the real one gives only now and then.
import AVFoundation
import Foundation
import Testing

private func page(_ total: Int, _ ids: [String]) -> String {
    let sessions = ids.map { #"{"id": "\#($0)", "name": "Session \#($0)"}"# }.joined(separator: ", ")
    return #"{"total": \#(total), "sessions": [\#(sessions)]}"#
}

private func dashboard(pending: String?) -> String {
    let pending = pending.map { #""\#($0)""# } ?? "null"
    return #"""
        {"name": "Session a", "problem": null, "pending": \#(pending), "summary": [], "failures": [], "caution": null,
         "highlights": [], "sections": []}
        """#
}

private func song(_ id: String, marked: [String] = ["REST", "FOCUS"]) -> String {
    let parts = marked.enumerated().map { index, mark in
        #"{"start_s": \#(index * 10), "end_s": \#(index * 10 + 10), "mark": "\#(mark)", "title": "\#(mark.capitalized)", "colour": "teal"}"#
    }
    return #"""
        {"id": "\#(id)", "title": "\#(id.capitalized)", "artist": null, "length_s": 60, "loudness": [0.5, 1],
         "parts": [\#(parts.joined(separator: ", "))], "cover": "7"}
        """#
}

/// Ten seconds of silence in a file of its own: a sound a player plays on, unheard.
func silence() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "silence-\(UUID().uuidString).caf")
    let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 8000, channels: 1))
    let frames = AVAudioFrameCount(8000 * 10)
    let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
    buffer.frameLength = frames  // zeros
    try AVAudioFile(forWriting: url, settings: format.settings).write(from: buffer)
    return url
}

private func songPage(_ songs: String...) -> String {
    #"{"total": \#(songs.count), "songs": [\#(songs.joined(separator: ", "))]}"#
}

private let marks = #"""
    {"unmarked": "Unmarked",
     "marks": [{"id": "REST", "title": "Rest", "colour": "teal"}, {"id": "FOCUS", "title": "Focus", "colour": "orange"}]}
    """#

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
            try await program.api.start(
                SessionRequest(sourceId: "s", recording: nil, songId: "one", name: "n", output: "o"))
        }
        await #expect(throws: APIError(message: "AdaMus did not answer as expected.", status: 500)) {
            try await program.api.runState()
        }
    }

    @Test func everyCallCarriesTheTokenAndASessionIsAskedForInTheWordsOfTheProgram() async throws {
        let program = StubProgram()
        program.answer("POST api/run", runState(1, "starting"))
        let request = SessionRequest(
            sourceId: "dataset", recording: "evening", songId: "Imagine", name: "Morning", output: "AirPods")
        _ = try await program.api.start(request)
        let call = try #require(program.made("POST api/run").first)
        #expect(call.token == program.api.token)
        let sent = try JSONSerialization.jsonObject(with: try #require(call.body)) as? [String: String]
        #expect(
            sent == [
                "source_id": "dataset", "recording": "evening", "song_id": "Imagine", "name": "Morning",
                "output": "AirPods",
            ])
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

    @Test func aRenamedSessionKeepsItsPlaceAndStaysChosenOnTheSamePart() async throws {
        let program = StubProgram()
        program.answer("GET api/sessions", page(2, ["a", "b"]))
        program.answer("PATCH api/sessions/b", #"{"id": "b-renamed", "name": "Karel 8"}"#)
        let workspace = Workspace(api: program.api)
        await workspace.load()
        workspace.selection = "b"
        workspace.session?.part = .details
        #expect(await workspace.rename(workspace.sessions[1], to: "Karel 8"))
        #expect(workspace.sessions.map(\.id) == ["a", "b-renamed"] && workspace.sessions[1].name == "Karel 8")
        #expect(workspace.selection == "b-renamed" && workspace.session?.summary.id == "b-renamed")
        #expect(workspace.session?.part == .details && workspace.problem == nil)
        let sent = try JSONSerialization.jsonObject(
            with: try #require(program.made("PATCH api/sessions/b").first?.body))
        #expect(sent as? [String: String] == ["name": "Karel 8"])
    }

    @Test func aSessionRenamedOnceAnotherWasChosenLeavesThatOneChosen() async {
        let program = StubProgram()
        program.answer("GET api/sessions", page(2, ["a", "b"]))
        program.answer("PATCH api/sessions/a", #"{"id": "a-renamed", "name": "New"}"#)
        let workspace = Workspace(api: program.api)
        await workspace.load()
        workspace.selection = "b"
        #expect(await workspace.rename(workspace.sessions[0], to: "New"))
        #expect(workspace.selection == "b" && workspace.sessions.map(\.id) == ["a-renamed", "b"])
    }

    @Test func aRefusedNameSaysWhyAndChangesNothing() async {
        let program = StubProgram()
        program.answer("GET api/sessions", page(1, ["a"]))
        program.answer("PATCH api/sessions/a", status: 422, #"{"detail": "Use at most 50 characters."}"#)
        let workspace = Workspace(api: program.api)
        await workspace.load()
        workspace.selection = "a"
        #expect(await !workspace.rename(workspace.sessions[0], to: "a/b"))
        #expect(workspace.problem == "Use at most 50 characters." && !workspace.isInFront)  // said in an alert
        #expect(workspace.selection == "a" && workspace.sessions.map(\.name) == ["Session a"])
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
struct LibraryTests {
    private func read(_ songs: String...) async -> (StubProgram, LibraryModel) {
        let program = StubProgram()
        program.answer("GET api/marks", marks)
        for answer in songs.isEmpty ? [songPage()] : songs { program.answer("GET api/songs", answer) }
        let library = LibraryModel(api: program.api)
        await library.load()
        return (program, library)
    }

    @Test func theSongsAreListedWithWhatAPartCanBeMarkedAndAFailedLookSaysWhy() async {
        let (program, library) = await read(songPage(song("one"), song("two")))
        #expect(library.songs.map(\.id) == ["one", "two"] && library.marks.map(\.id) == ["REST", "FOCUS"])
        #expect(library.unmarked == "Unmarked" && library.listProblem == nil)
        program.answerFromNow("GET api/songs", status: 500, #"{"detail": "The library cannot be read."}"#)
        await library.load()
        #expect(library.listProblem == "The library cannot be read." && library.songs.map(\.id) == ["one", "two"])
        #expect(program.made("GET api/marks").count == 1)  // they do not change
    }

    @Test func theSongsAreReadAPageAtATime() async {
        let (program, library) = await read(#"{"total": 3, "songs": [\#(song("a")), \#(song("b"))]}"#)
        // the second page was asked for, from where the first ended
        #expect(program.made("GET api/songs").map { $0.query["offset"] } == ["0", "2"])
        #expect(library.songs.map(\.id) == ["a", "b"])
    }

    @Test func aFileIsAnalysedIntoADraftAndOneRefusedSaysWhy() async throws {
        let (program, library) = await read()
        program.answer("POST api/drafts", song("d1", marked: []))
        program.answer("POST api/drafts", status: 422, #"{"detail": "notes.txt: not a sound file."}"#)
        await library.add(URL(filePath: "/music/one.mp3"))
        #expect(library.draft?.id == "d1" && library.draft?.of == nil && library.adding == nil)
        await library.discard()
        await library.add(URL(filePath: "/music/notes.txt"))
        #expect(library.draft == nil && library.problem == "notes.txt: not a sound file." && library.adding == nil)
        let sent = try JSONSerialization.jsonObject(with: try #require(program.made("POST api/drafts").first?.body))
        #expect(sent as? [String: String] == ["path": "/music/one.mp3"])
    }

    @Test func aStretchMarkedOrUnmarkedIsSentAndTheDraftComesBackAsTheProgramMarkedIt() async throws {
        let (program, library) = await read()
        program.answer("POST api/drafts", song("d1", marked: []))
        program.answer("POST api/drafts/d1/marks", song("d1", marked: ["REST"]))
        program.answer("POST api/drafts/d1/marks", song("d1", marked: []))
        await library.add(URL(filePath: "/music/one.mp3"))
        try await library.mark(1.5...9.0, as: "REST")
        #expect(library.draft?.song.parts.map(\.mark) == ["REST"])
        try await library.mark(0...10, as: nil)
        #expect(library.draft?.song.parts.isEmpty == true)
        let sent = try program.made("POST api/drafts/d1/marks").map {
            try JSONSerialization.jsonObject(with: try #require($0.body)) as? NSDictionary
        }
        #expect(sent == [["start_s": 1.5, "end_s": 9.0, "mark": "REST"], ["start_s": 0, "end_s": 10]])  // nil left out
    }

    @Test func aDraftKeptIsASongOfTheLibrarySelectedAsItWasCalled() async throws {
        let (program, library) = await read(songPage())
        program.answer("POST api/drafts", song("d1"))
        program.answer("POST api/drafts/d1/keep", song("one"))
        program.answerFromNow("GET api/songs", songPage(song("one")))
        await library.add(URL(filePath: "/music/one.mp3"))
        try await library.keep(title: "One", artist: "", picture: URL(filePath: "/pictures/front.png"))
        #expect(library.draft == nil && library.songs.map(\.id) == ["one"] && library.selection == "one")
        let sent = try JSONSerialization.jsonObject(
            with: try #require(program.made("POST api/drafts/d1/keep").first?.body))
        // no artist: it is not known
        #expect(sent as? [String: String] == ["title": "One", "picture": "/pictures/front.png"])
        await library.discard()  // the editor closes: nothing is put aside any more
        #expect(program.made("DELETE api/drafts/d1").isEmpty)
    }

    @Test func aDraftNotKeptStaysInTheEditorAndSaysWhy() async throws {
        let (program, library) = await read(songPage(song("one")))
        program.answer("POST api/songs/one/draft", song("d1"))
        program.answer("POST api/drafts/d1/keep", status: 422, #"{"detail": "A song has a title."}"#)
        await library.edit("one")
        await #expect(throws: APIError(message: "A song has a title.", status: 422)) {
            try await library.keep(title: " ", artist: "", picture: nil)
        }
        #expect(library.draft?.id == "d1" && library.draft?.of == "one")
    }

    @Test func aDraftPutAsideIsDeletedOnce() async {
        let (program, library) = await read(songPage(song("one")))
        program.answer("POST api/songs/one/draft", song("d1"))
        program.answer("DELETE api/drafts/d1", status: 204, "")
        await library.edit("one")
        await library.discard()
        await library.discard()
        #expect(library.draft == nil && program.made("DELETE api/drafts/d1").count == 1)
    }

    @Test func aDeletedSongLeavesTheLibraryAndAFailedDeletionSaysWhy() async throws {
        let (program, library) = await read(songPage(song("one"), song("two")), songPage(song("two")))
        program.answer("DELETE api/songs/one", status: 204, "")
        program.answer("DELETE api/songs/two", status: 404, #"{"detail": "No song two."}"#)
        library.selection = "one"
        library.askToDelete("one")
        await library.delete(try #require(library.toDelete))
        #expect(library.songs.map(\.id) == ["two"] && library.selection == nil && library.problem == nil)
        await library.delete(try #require(library.songs.first))
        #expect(library.problem == "No song two." && library.songs.map(\.id) == ["two"])
    }
}

@MainActor
struct SoundPlayerTests {
    @Test func oneSoundAtATimePlaysAndIsPutSomewhereAnotherShowsItsStart() {
        let player = SoundPlayer()
        // playing as soon as it is asked to: what it plays is not read before this test ends
        let one = FileManager.default.temporaryDirectory.appending(path: "one.wav")
        let other = FileManager.default.temporaryDirectory.appending(path: "other.wav")
        player.seek(to: 5, in: one)
        #expect(!player.plays(one) && player.time(of: one) == 5 && player.time(of: other) == 0)
        player.play(one)
        #expect(player.plays(one) && !player.plays(other))
        player.seek(to: 3, in: other)  // put somewhere, the other stops
        #expect(!player.plays(one) && player.time(of: one) == 0 && player.time(of: other) == 3)
        player.stop()
    }

    @Test func aStretchPlayedOverAndOverStopsTheSoundWhereItIsAndLeavesItThere() async throws {
        let player = SoundPlayer()
        defer { player.stop() }
        let sound = try silence()
        player.seek(to: 5, in: sound)
        player.play(sound)
        player.loop(1...1.3, in: sound)
        #expect(!player.playing && player.time(of: sound) == 5, "stopped where it was")
        let heard = try #require(player.loopTime(of: sound))
        #expect(player.loops(sound) && (1...1.3).contains(heard), "heard within the stretch")
        try await Task.sleep(for: .milliseconds(500))
        player.loop(0.5...1.3, in: sound)  // dragged on
        player.endLoop()
        #expect(!player.playing && player.time(of: sound) == 5, "and still there")
        #expect(!player.loops(sound) && player.loopTime(of: sound) == nil)
        player.play(sound)
        #expect(player.plays(sound) && player.time(of: sound) >= 5, "played on from there")
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

    @Test func whatAPartPlaysIsPausedAsItIsLeft() {
        let session = SessionModel(api: StubProgram().api, summary: SessionSummary(id: "a", name: "Session a"))
        defer { session.stop() }
        // playing as soon as it is asked to: what it plays is not read before this test ends
        let sound = FileManager.default.temporaryDirectory.appending(path: "sound.wav")
        session.music.play(sound)
        session.part = .overview
        #expect(session.music.playing, "the same part chosen again leaves it playing")
        session.part = .timeline
        #expect(!session.music.playing, "the overview left, its music is paused")
        session.music.play(sound)
        session.stop()
        #expect(!session.music.playing && session.music.time == 0, "another session chosen, it stops")
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
         "sources_hint": "No source.",
         "name_pattern": "[a-z]+", "name_hint": "Small letters only."}
        """#
    private static let fine = #"{"warning": null, "problem": null}"#
    private static let outputs = #"""
        {"outputs": [{"id": "TV", "title": "TV"}, {"id": "Speakers", "title": "Speakers"}],
         "system": "Speakers", "hint": "Connect headphones."}
        """#

    private static let songs = songPage(song("first"), song("second"))

    /// The form opened, and the song ``chosen`` chosen in the library, as its window reads it.
    private func opened(
        refresh: (status: Int, body: String)? = nil, songs: String = songs, chosen: Song.ID? = "first"
    ) async -> (StubProgram, NewSessionModel) {
        let program = StubProgram()
        program.answer("GET api/new-session", Self.form)
        let refresh = refresh ?? (200, Self.form)
        program.answer("POST api/new-session/refresh", status: refresh.status, refresh.body)
        program.answer("GET api/sources/headset/check", Self.fine)  // the first source, chosen at first
        program.answer("GET api/audio-outputs", Self.outputs)
        program.answer("GET api/marks", marks)
        program.answer("GET api/songs", songs)
        let model = NewSessionModel(api: program.api, library: LibraryModel(api: program.api))
        await model.open()
        if let chosen {
            await model.library.load()
            model.choose(song: chosen)
        }
        return (program, model)
    }

    @Test func theOutputTheSystemPlaysOnIsChosenAtFirstAndTheSessionIsAskedForOnTheOneChosen() async throws {
        let (program, model) = await opened()
        #expect(model.output?.id == "Speakers" && !model.lookingForOutputs)
        model.chosenOutput = "TV"
        model.name = "morning"
        program.answer("POST api/run", runState(1, "starting"))
        _ = await model.start()
        let sent = try JSONSerialization.jsonObject(with: try #require(program.made("POST api/run").first?.body))
        #expect((sent as? [String: String?])?["output"] == "TV")
    }

    @Test func noSongIsChosenAtFirstAndTheLibraryChoosesOneWhichTheSessionIsAskedFor() async throws {
        let (program, model) = await opened(chosen: nil)
        model.name = "morning"
        #expect(model.song == nil && !model.canStart)
        #expect(program.made("GET api/songs").isEmpty, "the library is read where it is shown")
        await model.library.load()
        model.askLibraryForSong()
        #expect(model.library.selection == nil)  // none chosen: none selected in the library
        model.library.choose?("second")
        #expect(model.song?.id == "second" && model.canStart)
        model.askLibraryForSong()
        #expect(model.library.selection == "second")  // the one chosen, selected again
        program.answer("POST api/run", runState(1, "starting"))
        _ = await model.start()
        let sent = try JSONSerialization.jsonObject(with: try #require(program.made("POST api/run").first?.body))
        #expect((sent as? [String: String?])?["song_id"] == "second")
    }

    @Test func aSongChosenThatIsDeletedIsChosenNoMoreAndNoSessionStarts() async {
        let (program, model) = await opened()
        model.choose(song: "second")
        model.name = "morning"
        program.answerFromNow("GET api/songs", songPage(song("first")))
        await model.library.load()
        #expect(model.song == nil && !model.canStart)  // not another in its place
        model.choose(song: "first")
        #expect(model.song?.id == "first" && model.canStart)
    }

    @Test func whatTheLibraryPlaysStopsAsTheSessionStarts() async throws {
        let (program, model) = await opened()
        model.library.player.play(try silence())  // a sound that plays on, unheard
        #expect(model.library.player.playing)
        model.name = "morning"
        program.answer("POST api/run", runState(1, "starting"))
        _ = await model.start()
        #expect(!model.library.player.playing)  // the session's own music is to be heard
    }

    @Test func aSongAndItsCoverAreAskedForWithTheTokenAndTheCoverWithItsVersionInTheirAddress() async throws {
        let (program, model) = await opened()
        let (sound, cover) = program.api.song(try #require(model.library.songs.first { $0.id == "second" }))
        let token = URLQueryItem(name: "token", value: program.api.token)
        let asked = [
            (sound, "/api/songs/second/sound", [token]),
            (try #require(cover), "/api/songs/second/cover", [token, URLQueryItem(name: "version", value: "7")]),
        ]
        for (url, path, query) in asked {
            let parts = URLComponents(url: url, resolvingAgainstBaseURL: false)
            #expect(parts?.path == path && parts?.queryItems == query)
        }
    }

    @Test func aRunWhoseSessionNeverStartedIsNotSaidToHaveEnded() async {
        let (_, model) = await opened()
        model.reportRun(problem: "Headphones is not connected.", started: false)
        #expect(model.problem == "Headphones is not connected." && !model.runEnded)
        model.dismissProblem()
        model.reportRun(problem: "The audio output was disconnected.", started: true)
        #expect(model.runEnded)
    }

    @Test func outputsThatCannotBeReadSayWhy() async {
        let (program, model) = await opened()
        program.answerFromNow("GET api/audio-outputs", status: 422, #"{"detail": "macOS did not tell."}"#)
        await model.lookForOutputs()
        #expect(model.problem == "macOS did not tell.")
    }

    @Test func anOutputChosenStaysChosenWhileItIsThereAndTheSystemsIsChosenOnceItIsGone() async {
        let (program, model) = await opened()
        model.chosenOutput = "TV"
        await model.lookForOutputs()
        #expect(model.output?.id == "TV")
        program.answerFromNow(
            "GET api/audio-outputs",
            #"{"outputs": [{"id": "AirPods", "title": "AirPods"}, {"id": "Speakers", "title": "Speakers"}], "system": "AirPods", "hint": ""}"#
        )
        await model.lookForOutputs()
        #expect(model.output?.id == "AirPods")
    }

    @Test func aMacWithoutOutputsOffersNoneAndNoSessionStarts() async {
        let (program, model) = await opened()
        program.answerFromNow(
            "GET api/audio-outputs", #"{"outputs": [], "system": null, "hint": "Connect headphones."}"#)
        await model.lookForOutputs()
        model.name = "morning"
        #expect(model.output == nil && model.outputs?.hint == "Connect headphones." && !model.canStart)
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
        let run = RunModel(
            api: program.api, state: RunState(run: 1, phase: .running, sessionId: nil, problem: nil))
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
        let run = RunModel(
            api: program.api, state: RunState(run: 1, phase: .running, sessionId: nil, problem: nil))
        await run.refresh()
        #expect(run.problem == "Busy." && !run.ended)  // said, and followed on
        await run.follow()
        #expect(run.outcome == .stored("s") && run.problem == nil)
    }

    @Test func aStopTheProgramRefusedCanBeAskedForAgain() async {
        let program = StubProgram()
        program.answer("POST api/run/stop", status: 500, #"{"detail": "Busy."}"#)
        program.answer("POST api/run/stop", status: 204, "")
        let run = RunModel(
            api: program.api, state: RunState(run: 1, phase: .running, sessionId: nil, problem: nil))
        await run.stop()
        #expect(!run.stopping && run.problem == "Busy.")
        await run.stop()
        #expect(run.stopping && program.made("POST api/run/stop").count == 2)
    }
}
