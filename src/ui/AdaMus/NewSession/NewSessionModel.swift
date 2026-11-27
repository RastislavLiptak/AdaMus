// The form of a new session: what the program offers, what has been chosen (the first source,
// the first recording of a dataset, unless the operator chose another: HIG "Pop-up buttons",
// a useful default), what the program says of the choice, and the name, checked by the
// program's own rule as it is typed. Opened, the form has the program look for the headset,
// which takes seconds: meanwhile the datasets are offered, and a session from one can start.
// The music plays on the audio output chosen, at first the one the system plays on. Its source,
// a song, is chosen in the song library, none until the operator chooses one; what the library
// plays stops as the session starts, whose own music is to be heard.
import Foundation
import Observation

@MainActor @Observable
final class NewSessionModel {
    let api: APIClient
    let library: LibraryModel
    private(set) var form: NewSessionForm?
    private(set) var lookingForHeadset = false
    /// The source the operator chose; until then, the first one offered.
    var chosenSource: SourceChoice.ID? {
        didSet { if chosenSource != oldValue { chosenRecording = nil } }
    }
    /// The recording the operator chose; until then, the first of the source.
    var chosenRecording: String?
    /// The song the operator chose; none until then.
    private(set) var chosenSong: Song.ID?
    var name = ""
    private(set) var check: SourceCheck?
    private(set) var outputs: AudioOutputs?
    /// The output the operator chose; until then, the one the system plays on.
    var chosenOutput: AudioOutputChoice.ID?
    private(set) var lookingForOutputs = false
    /// Why the program refused what was asked of it, or why a session it ran did not end well.
    private(set) var problem: String?
    /// Whether ``problem`` is of a session that ran, not of one that never started.
    private(set) var runEnded = false
    private(set) var starting = false

    init(api: APIClient, library: LibraryModel) {
        self.api = api
        self.library = library
    }

    var source: SourceChoice? {
        guard let sources = form?.sources else { return nil }
        return sources.first { $0.id == chosenSource } ?? sources.first
    }

    /// The song chosen, while the library has it: a song deleted meanwhile is chosen no more.
    var song: Song? {
        library.songs.first { $0.id == chosenSong }
    }

    var output: AudioOutputChoice? {
        guard let outputs else { return nil }
        let chosen = outputs.outputs.first { $0.id == chosenOutput }
        return chosen ?? outputs.outputs.first { $0.id == outputs.system } ?? outputs.outputs.first
    }

    var recording: String? {
        guard let recordings = source?.recordings, !recordings.isEmpty else { return nil }
        return recordings.first { $0 == chosenRecording } ?? recordings.first
    }

    /// What is wrong with the name; nothing while none has been typed.
    var nameProblem: String? {
        guard let form, !name.isEmpty else { return nil }
        return Self.isName(name, pattern: form.namePattern) ? nil : form.nameHint
    }

    /// The headset is a source of its own: none of its sessions starts while it is looked for.
    var waitsForHeadset: Bool { lookingForHeadset && source?.live == true }

    var canStart: Bool {
        guard let form, source != nil, song != nil, !name.isEmpty, !starting else { return false }
        return Self.isName(name, pattern: form.namePattern) && check?.problem == nil && !waitsForHeadset
            && output != nil
    }

    /// The form as the program offers it, then again once it has looked for the headset; the
    /// audio outputs meanwhile.
    func open() async {
        async let outputsRead: Void = lookForOutputs()
        await lookForSources()
        await outputsRead
    }

    /// The form as the program offers it, then again once it has looked for the headset.
    func lookForSources() async {
        do {
            form = try await api.newSessionForm()  // its choice is checked by the form as shown
            lookingForHeadset = true
            defer { lookingForHeadset = false }
            form = try await api.lookForHeadset()
            await checkChoice()  // the choice may stay, and the form check it no more
        } catch let refused as APIError where refused.status == 409 {
            return  // the headset is looked for already, or a session runs: the form stands
        } catch {
            problem = error.localizedDescription
        }
    }

    /// What the program says of the source and recording chosen.
    func checkChoice() async {
        guard let source else {
            check = nil
            return
        }
        let asked = (source.id, recording)
        do {
            let checked = try await api.check(source: source.id, recording: recording)
            if !Task.isCancelled, self.source?.id == asked.0, self.recording == asked.1 {  // not stale
                check = checked
            }
        } catch is CancellationError {
            return
        } catch let failed as URLError where failed.code == .cancelled {
            return
        } catch {
            problem = error.localizedDescription
        }
    }

    /// The audio outputs the Mac has now. The one chosen stays chosen while it is there.
    func lookForOutputs() async {
        lookingForOutputs = true
        defer { lookingForOutputs = false }
        do {
            outputs = try await api.audioOutputs()
        } catch {
            problem = error.localizedDescription
        }
    }

    func choose(song id: Song.ID) {
        chosenSong = id
    }

    /// Have the song library choose the song, the one chosen selected in it at first.
    func askLibraryForSong() {
        library.selection = song?.id
        library.choose = { [weak self] in self?.choose(song: $0) }
    }

    /// Ask for the session; the state of its run, once the program has taken it.
    func start() async -> RunState? {
        guard canStart, let source, let song, let output else { return nil }
        library.player.stop()  // the session's own music is to be heard
        starting = true
        defer { starting = false }
        do {
            let request = SessionRequest(
                sourceId: source.id, recording: recording, songId: song.id, name: name, output: output.id)
            return try await api.start(request)
        } catch {
            problem = error.localizedDescription
            return nil
        }
    }

    func dismissProblem() {
        problem = nil
        runEnded = false
    }

    /// A run that did not end well says why, here; ``started`` whether its session ran at all.
    func reportRun(problem: String, started: Bool) {
        self.problem = problem
        runEnded = started
    }

    /// Whether the whole ``name`` keeps the rule ``pattern`` the program gives.
    static func isName(_ name: String, pattern: String) -> Bool {
        guard let rule = try? NSRegularExpression(pattern: pattern) else { return false }
        let whole = NSRange(name.startIndex..., in: name)
        return rule.firstMatch(in: name, options: .anchored, range: whole)?.range == whole
    }
}
