// The songs of the library, as its window works with them: read from the program a page at a
// time, the one selected, deleted after a question; and the draft where a song is changed, in
// the editor: of a file added (which the program analyses, seconds a song), which goes into the
// library once it is marked, or of a song of the library, which it then takes the place of. The
// form of a new session chooses its song here too: while it asks, the window chooses the song
// selected for it. One song plays at a time, wherever it is shown.
import Foundation
import Observation

@MainActor @Observable
final class LibraryModel {
    let api: APIClient
    private(set) var songs: [Song] = []
    /// Why the songs could not be read, if they could not.
    private(set) var listProblem: String?
    /// The states a stretch can be marked, as the program names and colours them.
    private(set) var marks: [MarkChoice] = []
    /// What a part no mark covers is called.
    private(set) var unmarked = ""
    /// The name of the file being analysed, until its draft is ready.
    private(set) var adding: String?
    /// The song being changed in the editor.
    private(set) var draft: Draft?
    var selection: Song.ID?
    /// The song whose deletion waits to be confirmed.
    var toDelete: Song?
    /// What went wrong with something asked for in the window, to be said in an alert.
    var problem: String?
    /// Takes the song chosen, while the form of a new session asks for one.
    var choose: ((Song.ID) -> Void)?
    let player = SoundPlayer()

    /// A song being changed, and the song of the library it is a draft of; none of a file added.
    struct Draft: Identifiable {
        var song: Song
        let of: Song.ID?
        var id: Song.ID { song.id }
    }

    init(api: APIClient) {
        self.api = api
    }

    /// No song is being analysed or changed: one song at a time.
    var canAdd: Bool {
        adding == nil && draft == nil
    }

    var selected: Song? {
        songs.first { $0.id == selection }
    }

    /// The songs again; the one selected stays selected while it is there.
    func load() async {
        do {
            if marks.isEmpty {
                let known = try await api.songMarks()
                (marks, unmarked) = (known.marks, known.unmarked)
            }
            songs = try await api.songs()
            listProblem = nil
            if selected == nil { selection = nil }
        } catch {
            listProblem = error.localizedDescription
        }
    }

    /// Change the song in the file at ``url``, which goes into the library once it is marked; what
    /// the program refused, an alert says why.
    func add(_ url: URL) async {
        adding = url.deletingPathExtension().lastPathComponent
        defer { adding = nil }
        do {
            draft = Draft(song: try await api.draft(ofFile: url.path(percentEncoded: false)), of: nil)
        } catch {
            problem = error.localizedDescription
        }
    }

    /// Change the song ``song`` of the library.
    func edit(_ song: Song.ID) async {
        do {
            draft = Draft(song: try await api.draft(of: song), of: song)
        } catch {
            problem = error.localizedDescription
        }
    }

    /// Mark the stretch ``range`` of the draft as ``mark``; nil unmarks it.
    func mark(_ range: ClosedRange<Double>, as mark: String?) async throws {
        guard let id = draft?.id else { return }
        let marked = try await api.mark(id, SongMark(startS: range.lowerBound, endS: range.upperBound, mark: mark))
        if draft?.id == id { draft?.song = marked }
    }

    /// Put the draft into the library, called ``title``, by ``artist`` (not known while empty),
    /// with the picture in the file at ``picture`` if one was chosen; its song is then selected.
    func keep(title: String, artist: String, picture: URL?) async throws {
        guard let id = draft?.id else { return }
        let kept = try await api.keep(
            id,
            as: DraftKept(
                title: title, artist: artist.isEmpty ? nil : artist, picture: picture?.path(percentEncoded: false)))
        draft = nil
        await load()
        selection = kept.id
    }

    /// Put the draft aside, as it was never changed.
    func discard() async {
        guard let id = draft?.id else { return }
        draft = nil
        try? await api.discard(id)  // one the program has lost already is put aside as it opens again
    }

    func askToDelete(_ id: Song.ID?) {
        toDelete = songs.first { $0.id == id }
    }

    func delete(_ song: Song) async {
        player.stop()  // it may play from the file deleted
        do {
            try await api.deleteSong(song.id)
            if selection == song.id { selection = nil }
            await load()
        } catch {
            problem = error.localizedDescription
        }
    }
}
