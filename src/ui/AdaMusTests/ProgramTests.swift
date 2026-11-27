// The app with the program. The real program, on the stand-in computer of the tests
// (tests/support/stand_in_program.py), is started by the Python test that runs these tests
// (tests/app/test_app.py), which gives them what it printed and where its API answers: every
// step of the app is taken through that API, so that the app and the program cannot drift
// apart. (The app does not start that program here: the project lies in a folder macOS guards,
// which an app built for a test may not read unasked.) How the app starts a program, follows it
// and ends it, small programs of the tests show. Run alone, from Xcode, the tests of the real
// program are left out.
import AVFoundation
import Foundation
import Testing

/// What the Python test gives: the lines the program printed, and where its API answers.
enum RealProgram {
    static let environment = ProcessInfo.processInfo.environment
    static var given: Bool {
        ["ADAMUS_MESSAGES", "ADAMUS_ADDRESS", "ADAMUS_TOKEN", "ADAMUS_ANSWERS"].allSatisfy { environment[$0] != nil }
    }

    static var messages: [String] { (environment["ADAMUS_MESSAGES"] ?? "").split(separator: "\n").map(String.init) }

    static var api: APIClient {
        APIClient(
            address: URL(string: environment["ADAMUS_ADDRESS"] ?? "http://127.0.0.1")!,
            token: environment["ADAMUS_TOKEN"] ?? "")
    }
}

@MainActor
@Suite(.enabled(if: RealProgram.given), .serialized)
struct RealProgramTests {
    @Test func everyLineTheProgramPrintedIsAMessageTheLastWhereItsApiAnswers() {
        let messages = RealProgram.messages.map { ProgramMessage(line: $0) }
        #expect(messages.count > 2 && messages.allSatisfy { $0 != nil })
        guard case .checks(let first) = messages.first, case .ready(let address, let token) = messages.last else {
            Issue.record("the checklist first, where the API answers last")
            return
        }
        #expect(first.allSatisfy { $0.status == .pending })
        #expect(address == RealProgram.api.address && token == RealProgram.api.token)
    }

    @Test func theSessionsTheirDashboardTheirTimelineTheirSamplesAndSound() async throws {
        let api = RealProgram.api
        // as tests/app played it
        let played = try #require(try await api.sessions().first { $0.name == "morning_focus" })

        let dashboard = try await api.dashboard(of: played.id)
        #expect(dashboard.name == played.name)
        #expect(!dashboard.highlights.isEmpty && !dashboard.sections.isEmpty)
        #expect(!dashboard.summary.isEmpty)

        let timeline = try await api.timeline(of: played.id)
        for track in timeline.groups.flatMap(\.tracks) {
            switch track {
            case .signal(let signal):
                let samples = try await api.samples(of: signal.id, in: played.id)
                #expect(samples.count == signal.samples * signal.layers)
            case .audio(let audio):
                let envelope = try await api.samples(of: audio.id, in: played.id)
                #expect(envelope.count == 2 * audio.buckets)
                let (sound, answer) = try await URLSession.shared.data(from: api.soundURL(of: audio.id, in: played.id))
                #expect((answer as? HTTPURLResponse)?.statusCode == 200 && sound.count > 44)  // more than a header
            case .curve:
                continue
            }
        }
        let model = TimelineModel(timeline: timeline) { api.soundURL(of: $0, in: played.id) }
        try await model.load { try await api.samples(of: $0, in: played.id) }
        let sampled = timeline.groups.flatMap(\.tracks).filter {
            if case .signal = $0 { true } else if case .audio = $0 { true } else { false }
        }
        #expect(Set(model.pyramids.byTrack.keys) == Set(sampled.map(\.id)))  // each drawn, none left out
        for pyramid in model.pyramids.byTrack.values.joined() {
            #expect(zip(pyramid.samples.low, pyramid.samples.high).allSatisfy { !($0 > $1) })
        }
    }

    @Test func theHeadAndTheElectrodesOfASessionOnItAreReadFromTheProgram() async throws {
        let api = RealProgram.api
        let played = try #require(try await api.sessions().first { $0.name == "morning_focus" })
        let head = try await api.head()
        #expect(head.vertices.count % 3 == 0 && head.triangles.count % 3 == 0 && !head.triangles.isEmpty)
        #expect(Int(head.triangles.max() ?? 0) < head.vertices.count / 3)
        let electrodes = try await api.electrodes(of: played.id)
        #expect(electrodes.last?.name == "Fpz" && electrodes.last?.kind == .reference)  // a Muse 2
        #expect(
            electrodes.filter { $0.kind == .recorded }.allSatisfy { $0.position?.count == 3 && $0.normal?.count == 3 })
        #expect(electrodes.allSatisfy { $0.about.hasPrefix($0.name) })
        let set = ElectrodeSet(head: head, electrodes: electrodes)
        #expect(set.placed.count == electrodes.count && set.colours.count == electrodes.count)
    }

    @Test func aPartOfTheTimelineIsWrittenOutByTheProgram() async throws {
        let api = RealProgram.api
        let played = try #require(try await api.sessions().first { $0.name == "morning_focus" })
        let excerpt = try await api.excerpt(of: played.id, from: 10, to: 20)
        let data = try #require(try JSONSerialization.jsonObject(with: Data(excerpt.text.utf8)) as? [String: Any])
        #expect((data["session"] as? [String: Any])?["name"] as? String == "morning_focus")
        #expect((data["part"] as? [String: Any])?["start_shown"] as? String == "0:10.0")
        await #expect(throws: APIError.self) { try await api.excerpt(of: played.id, from: 20, to: 10) }
    }

    /// What the window reads that the session played for the tests has: each value it may lack
    /// is there somewhere, read under the name the program gives it.
    @Test func everyValueOfTheTimelineAndDashboardThatMayLackIsReadWhereTheProgramGivesIt() async throws {
        let api = RealProgram.api
        let played = try #require(try await api.sessions().first { $0.name == "morning_focus" })
        let timeline = try await api.timeline(of: played.id)
        let tracks = timeline.groups.flatMap(\.tracks)
        let curves = tracks.compactMap { if case .curve(let track) = $0 { track } else { nil } }
        let bands = curves.flatMap(\.bands)
        #expect(tracks.contains { $0.caption != nil })
        #expect(bands.contains { $0.title != nil } && bands.contains { $0.colour != nil })
        #expect(curves.contains { $0.reference != nil && $0.above != nil && $0.below != nil })
        let steps = curves.flatMap(\.steps)
        #expect(steps.contains { $0.value != nil })  // a colour only where a step was rejected: none in it
        #expect(timeline.consoles.flatMap(\.lines).contains { $0.source != nil })

        let dashboard = try await api.dashboard(of: played.id)
        #expect(dashboard.summary.contains { $0.status == .good })
    }

    /// What the program says only now and then, made by its own models (tests/app/test_app.py),
    /// each value set to its own name.
    @Test func whatTheProgramSaysOnlyNowAndThenIsReadUnderTheNamesItGives() throws {
        struct Answers: Decodable {
            let dashboard: Dashboard
            let sourceCheck: SourceCheck
            let runState: RunState
        }
        let given = try #require(RealProgram.environment["ADAMUS_ANSWERS"])
        let answers = try APIClient.decoder.decode(Answers.self, from: Data(given.utf8))
        #expect(answers.dashboard.problem == "problem" && answers.dashboard.pending == "pending")
        #expect(answers.dashboard.failures == ["failure"] && answers.dashboard.caution == "caution")
        #expect(answers.sourceCheck == SourceCheck(warning: "warning", problem: "problem"))
        #expect(answers.runState == RunState(run: 1, phase: .idle, sessionId: "session_id", problem: "problem"))
    }

    @Test func aSessionWithTheHeadsetRunsUntilItIsStoppedAndIsDeletedAgain() async throws {
        let api = RealProgram.api
        let form = NewSessionModel(api: api, library: LibraryModel(api: api))
        await form.open()
        #expect(form.source?.live == true && form.recording == nil)  // the headset has no recordings
        await form.library.load()
        form.choose(song: try #require(form.library.songs.first).id)
        form.name = "live"
        let started = try #require(await form.start())
        #expect(form.problem == nil)
        let run = RunModel(api: api, state: started)
        let end = ContinuousClock.now + .seconds(60)
        while run.state.phase != .running && !run.ended {
            guard ContinuousClock.now < end else { throw APIError(message: "never ran", status: nil) }
            try await Task.sleep(for: .milliseconds(50))
            await run.refresh()
        }
        await run.stop()
        await run.follow()
        #expect(run.problem == nil && run.state.problem == nil)
        let id = try #require(run.state.sessionId)
        try await api.delete(id)
    }

    @Test func aCallWithoutTheTokenFindsNothing() async throws {
        let stranger = APIClient(address: RealProgram.api.address, token: "guess")
        await #expect(throws: APIError.self) { try await stranger.sessions() }
    }

    @Test func theLibraryIsReadAndASongIsChangedInADraftThenGivenBack() async throws {
        let api = RealProgram.api
        let library = LibraryModel(api: api)
        await library.load()
        #expect(library.listProblem == nil && library.marks.map(\.title) == ["Rest", "Focus"])
        #expect(library.unmarked == "Unmarked")
        let song = try #require(library.songs.first { $0.title == "Stand-in Song" })
        #expect(song.parts.map(\.title) == ["Rest", "Focus"])
        // marked where it is put, in a draft, which leaves the song as it is until it is kept
        await library.edit(song.id)
        try await library.mark(1.2...3.9, as: "FOCUS")
        let marked = try #require(library.draft?.song)
        #expect(marked.parts.map(\.title) == ["Rest", "Focus", "Rest", "Focus"])
        #expect(marked.parts[1].startS == 1.2 && marked.parts[1].endS == 3.9)
        let (draftSound, _) = api.draftFiles(marked)
        let draftLength = try await AVURLAsset(url: draftSound).load(.duration).seconds
        #expect(abs(draftLength - song.lengthS) < 0.1)
        await library.discard()
        await library.load()
        #expect(library.songs.first { $0.id == song.id }?.parts == song.parts)
        // what is not a song is refused, as the program says why
        await library.add(try silence())
        #expect(library.draft == nil && library.problem?.contains("no beat") == true)
        // kept under another title and artist, then given back as it was
        await library.edit(song.id)
        try await library.keep(title: "Tokyo", artist: "The Clause", picture: nil)
        #expect(library.selected.map { [$0.title, $0.artist] } == ["Tokyo", "The Clause"])
        #expect(library.selected?.cover == song.cover && library.selected?.parts == song.parts)  // its own picture
        await library.edit(song.id)
        try await library.keep(title: song.title, artist: "", picture: nil)
        // its sound and picture, as a player and an image ask for them, without a header
        let (sound, cover) = api.song(song)
        let length = try await AVURLAsset(url: sound).load(.duration).seconds
        #expect(abs(length - song.lengthS) < 0.1)
        let (picture, given) = try await URLSession.shared.data(from: try #require(cover))
        #expect((given as? HTTPURLResponse)?.statusCode == 200 && picture.starts(with: [0x89, 0x50, 0x4E, 0x47]))
    }

    @Test func aNewSessionFromTheFormRunsToItsEndAndIsDeletedAgain() async throws {
        let api = RealProgram.api
        let before = try await api.sessions().count

        let form = NewSessionModel(api: api, library: LibraryModel(api: api))
        await form.open()
        #expect(form.problem == nil)
        #expect(form.form?.sources.first?.live == true)  // the stand-in headset, found and first
        // the stand-in Mac plays on its speakers; the session is asked for on its TV
        #expect(form.outputs?.outputs.map(\.id) == ["TV", "MacBook Pro Speakers"])
        #expect(form.output?.id == "MacBook Pro Speakers")
        form.chosenOutput = "TV"
        // no song until one is chosen in the library, played as it is asked for without a header
        #expect(form.song == nil && !form.canStart)
        await form.library.load()
        let song = try #require(form.library.songs.first { $0.title == "Stand-in Song" })
        form.choose(song: song.id)
        let (sound, answer) = try await URLSession.shared.data(from: api.song(song).sound)
        #expect((answer as? HTTPURLResponse)?.statusCode == 200 && sound.count > 100_000)
        #expect(sound.prefix(4) == Data("RIFF".utf8))  // a WAV, which the player and the program read
        form.chosenSource = form.form?.sources.first { !$0.live }?.id
        await form.checkChoice()
        #expect(form.recording != nil && form.check?.problem == nil)
        let rule = try #require(form.form?.namePattern)  // the program's own, as ICU reads it
        let names = [
            ("Karel Jedlička 8", true), ("  ok  ", true), ("👩‍🔬 test", true), (String(repeating: "x", count: 50), true),
            (" ", false), ("a/b", false), ("a:b", false), ("tab\there", false),
            (String(repeating: "x", count: 51), false),
        ]
        for (name, kept) in names { #expect(NewSessionModel.isName(name, pattern: rule) == kept, "\(name)") }
        form.name = "a/b"
        #expect(form.nameProblem == form.form?.nameHint && !form.canStart)
        form.name = "Karel Jedlička 8"
        #expect(form.nameProblem == nil && form.canStart)

        let started = try #require(await form.start())
        let run = RunModel(api: api, state: started)
        await run.follow()
        #expect(run.problem == nil && run.state.problem == nil)
        let id = try #require(run.state.sessionId)
        // its overview has the song it kept, as the library had it, which a player and an image
        // take as they are
        let source = try #require(try await api.dashboard(of: id).song)
        #expect(source.title == "Stand-in Song" && source.cover != nil && source.parts == song.parts)
        let (kept, cover) = api.song(ofSession: id, source)
        let length = try await AVURLAsset(url: kept).load(.duration).seconds
        #expect(abs(length - source.lengthS) < 0.1)
        // played from where it was put, to its end, then again from its beginning
        let player = SoundPlayer()
        player.seek(to: source.lengthS - 0.3, in: kept)
        player.play(kept)
        for _ in 0..<50 where player.playing { try await Task.sleep(for: .milliseconds(100)) }
        #expect(!player.playing && player.time == 0, "played to its end within 5 s, from where it was put")
        player.play(kept)
        for _ in 0..<20 where player.time == 0 { try await Task.sleep(for: .milliseconds(100)) }
        #expect(player.playing && player.time > 0 && player.time < 2.5, "and again from its beginning")
        // put elsewhere as it plays, it is there at once, not where it was until the seek lands
        player.seek(to: 10, in: kept)
        #expect(player.time == 10)
        // its clock goes once the audio output plays, which takes it a moment to wake up
        for _ in 0..<50 where player.time <= 10 { try await Task.sleep(for: .milliseconds(100)) }
        #expect(player.time > 10 && player.time < 11, "and plays on from there")
        // paused, it stays where it stopped
        player.pause()
        let paused = player.time
        try await Task.sleep(for: .milliseconds(300))
        #expect(!player.playing && paused > 10 && player.time == paused, "paused where it stopped")
        player.stop()
        let (picture, given) = try await URLSession.shared.data(from: try #require(cover))
        #expect((given as? HTTPURLResponse)?.statusCode == 200 && picture.starts(with: [0x89, 0x50, 0x4E, 0x47]))
        // listed under the name it was given (the stand-in computer's clock starts in 1970, so
        // which session is newest says nothing here)
        let listed = try await api.sessions()
        #expect(listed.contains(SessionSummary(id: id, name: "Karel Jedlička 8")))

        let workspace = Workspace(api: api)
        await workspace.load()
        await workspace.delete(try #require(workspace.sessions.first { $0.id == id }))
        #expect(workspace.problem == nil)
        #expect(try await api.sessions().count == before)
    }
}

/// Small programs that print what the program would, written for a test into a folder of its own.
@MainActor
enum SmallProgram {
    static func make(_ script: String) throws -> Program {
        let folder = try newFolder()
        let file = try write(script, to: folder.appending(path: "program"))
        return Program(executable: file, folder: folder)
    }

    static func newFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    /// A shell script at ``file``, which may be run.
    static func write(_ script: String, to file: URL) throws -> URL {
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try ("#!/bin/sh\n" + script).write(to: file, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: file.path(percentEncoded: false))
        return file
    }

    static func until(_ program: Program, _ holds: (Program.State) -> Bool) async throws {
        let end = ContinuousClock.now + .seconds(10)
        while !holds(program.state) {
            guard ContinuousClock.now < end else { throw APIError(message: "waited in vain", status: nil) }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    static let checks =
        #"{"checks": [{"name": "Configuration", "status": "passed", "detail": "config.yaml", "fix": null}, {"name": "Audio output", "status": "failed", "detail": "44100 Hz", "fix": "Set it to 48000 Hz."}]}"#
    static let ready = #"{"ready": {"address": "http://127.0.0.1:9", "token": "secret"}}"#
}

@MainActor
struct ProgramTests {
    @Test func aCheckThatFailedIsShownWithHowToFixItOnceTheProgramHasEnded() async throws {
        let program = try SmallProgram.make("echo '\(SmallProgram.checks)'\nexit 1\n")
        program.start()
        try await SmallProgram.until(program) { if case .failed = $0 { true } else { false } }
        guard case .failed(let checks) = program.state else { return }
        #expect(checks.map(\.status) == [.passed, .failed])
        #expect(checks.last?.fix == "Set it to 48000 Hz.")
    }

    @Test func readyOnceItSaysWhereItsApiAnswers() async throws {
        let program = try SmallProgram.make("echo '\(SmallProgram.ready)'\ncat > /dev/null\n")
        program.start()
        try await SmallProgram.until(program) { if case .ready = $0 { true } else { false } }
        #expect(program.state == .ready(APIClient(address: URL(string: "http://127.0.0.1:9")!, token: "secret")))
    }

    @Test func aProgramThatEndsOnItsOwnSaysWhyInItsLastLines() async throws {
        let program = try SmallProgram.make(
            "echo 'Traceback (most recent call last):' >&2\necho 'KeyError: x' >&2\nexit 3\n")
        program.start()
        try await SmallProgram.until(program) { if case .ended = $0 { true } else { false } }
        #expect(program.state == .ended("Traceback (most recent call last):\nKeyError: x\nAdaMus ended with status 3."))
        program.start()  // and starts again
        try await SmallProgram.until(program) { if case .ended = $0 { true } else { false } }
    }

    @Test func aProgramThatSaysNothingEndsWithItsStatus() async throws {
        let program = try SmallProgram.make("exit 4\n")
        program.start()
        try await SmallProgram.until(program) { if case .ended = $0 { true } else { false } }
        #expect(program.state == .ended("AdaMus ended with status 4."))
    }

    @Test func ofALongErrorTheLastLinesAreKeptAsTheCauseComesLast() async throws {
        let program = try SmallProgram.make("for i in $(seq 1 50); do echo \"line $i\" >&2; done\nexit 1\n")
        program.start()
        try await SmallProgram.until(program) { if case .ended = $0 { true } else { false } }
        let kept = (50 - Program.errorLinesKept + 1...50).map { "line \($0)" } + ["AdaMus ended with status 1."]
        #expect(program.state == .ended(kept.joined(separator: "\n")))
    }

    @Test func aProgramTheSystemStoppedSaysSoNotJustANumber() async throws {
        let program = try SmallProgram.make("echo 'looking for the headset' >&2\nkill -ABRT $$\n")
        program.start()
        try await SmallProgram.until(program) { if case .ended = $0 { true } else { false } }
        #expect(
            program.state
                == .ended(
                    "looking for the headset\nAdaMus was stopped by the system: \(String(cString: strsignal(SIGABRT)))."
                )
        )
    }

    @Test func aProgramThatRunsIsNotStartedASecondTime() async throws {
        let program = try SmallProgram.make("echo '\(SmallProgram.ready)'\ncat > /dev/null\n")
        program.start()
        try await SmallProgram.until(program) { if case .ready = $0 { true } else { false } }
        program.start()  // as the window does when it appears again
        #expect(program.state == .ready(APIClient(address: URL(string: "http://127.0.0.1:9")!, token: "secret")))
    }

    @Test func theAppStartsTheProgramOfTheProjectFolderInThatFolder() async throws {
        let project = try SmallProgram.newFolder()
        _ = try SmallProgram.write(
            "[ -d .venv ] || exit 5\necho '\(SmallProgram.ready)'\ncat > /dev/null\n",
            to: project.appending(path: ".venv/bin/adamus"))
        let program = Program.ofProject(project)
        program.start()
        try await SmallProgram.until(program) { if case .ready = $0 { true } else { false } }
    }

    @Test func anAppOutsideItsProjectFolderSaysSoAndStartsNothing() {
        let folder = FileManager.default.temporaryDirectory
        let program = Program.ofProject(folder)
        program.start()
        #expect(program.state == .notInProject(folder))
    }
}
