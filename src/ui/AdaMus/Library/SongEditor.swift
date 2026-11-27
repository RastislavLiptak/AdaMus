// A song changed, in a sheet over the song library, as a new session is made in one (HIG
// "Sheets"): what it is (its cover, chosen from a picture, its title and its artist) in a
// grouped form, then its parts, on its sound track, large, zoomed in to be exact with the
// buttons beside it or a pinch, and played and paused there. A stretch is selected along the
// track, heard as it is dragged, and marked from the menu over it (SoundTrack). A song added
// comes here first, and goes into the library with Add to Library; a song of the library takes
// its changes with Done; either once it has a title and a part marked each state. Cancel leaves
// it as it was.
import SwiftUI
import UniformTypeIdentifiers

struct SongEditor: View {
    let draft: LibraryModel.Draft
    let library: LibraryModel
    @State private var title: String
    @State private var artist: String
    /// The picture chosen for its cover, if one was.
    @State private var picture: URL?
    @State private var choosingPicture = false
    @State private var zoom = 1.0
    /// The zoom as a pinch began, while one goes on.
    @State private var pinched: Double?
    @State private var keeping = false
    /// What went wrong with something asked for, to be said in an alert.
    @State private var problem: String?
    @Environment(\.dismiss) private var dismiss

    init(draft: LibraryModel.Draft, library: LibraryModel) {
        self.draft = draft
        self.library = library
        _title = State(initialValue: draft.song.title)
        _artist = State(initialValue: draft.song.artist ?? "")
    }

    var body: some View {
        let (sound, cover) = library.api.draftFiles(draft.song)
        Form {
            Section {
                HStack(spacing: Metrics.Spacing.regular) {  // its label by its middle, as a picture is tall
                    Text("Cover")
                    Spacer()
                    Cover(picture: picture ?? cover, side: Metrics.Sound.thumbnail)
                    Button("Choose…") { choosingPicture = true }
                }
                TextField("Title", text: $title, prompt: Text("Required"))
                    .accessibilityLabel("Title")  // the form shows its title beside it, unread
                TextField("Artist", text: $artist, prompt: Text(unknownArtist))
                    .accessibilityLabel("Artist")
            }
            Section {
                SoundPosition(
                    shape: draft.song, sound: sound, player: library.player, name: "song",
                    marking: Marking(choices: library.marks, mark: mark, hear: { hear($0, sound) }),
                    zoom: zoom, height: Metrics.Sound.markingTrack
                )
                .simultaneousGesture(pinch)
                HStack {
                    controls(sound)
                    Spacer()
                    PartsLegend(shape: draft.song, unmarked: library.unmarked)
                }
            } header: {
                Text("Parts")
            } footer: {
                Text("Drag along the track to mark a part rest or focus. A song needs a part of each.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .navigationTitle(draft.of == nil ? "New Song" : "Edit Song")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel", role: .cancel) { dismiss() }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button(draft.of == nil ? "Add to Library" : "Done", action: keep)
                    .disabled(!canKeep)
            }
        }
        .fileImporter(isPresented: $choosingPicture, allowedContentTypes: [.jpeg, .png]) {
            if case .success(let url) = $0 { picture = url }
        }
        .alert(
            "The Request Could Not Be Done",
            isPresented: Binding(get: { problem != nil }, set: { if !$0 { problem = nil } })
        ) {
            Button("OK") {}
        } message: {
            Text(problem ?? "")
        }
        .frame(width: Metrics.songEditor)
        .onDisappear { library.player.stop() }  // its file goes with the draft
    }

    /// Play and pause, and zoom out and in, as the timeline of a session does.
    private func controls(_ sound: URL) -> some View {
        HStack(spacing: Metrics.Spacing.small) {
            let playing = library.player.plays(sound)
            Button(playing ? "Pause" : "Play", systemImage: playing ? "pause.fill" : "play.fill") {
                if playing { library.player.pause() } else { library.player.play(sound) }
            }
            .help(playing ? "Pause the Song" : "Play the Song")
            Button("Zoom Out", systemImage: "minus.magnifyingglass") { zoom = max(1, zoom / Metrics.Sound.zoomStep) }
                .disabled(zoom <= 1)
            Button("Zoom In", systemImage: "plus.magnifyingglass") {
                zoom = min(Metrics.Sound.mostZoom, zoom * Metrics.Sound.zoomStep)
            }
            .disabled(zoom >= Metrics.Sound.mostZoom)
        }
        .labelStyle(.iconOnly)
    }

    /// It has a title, and each state the program knows is marked somewhere.
    private var canKeep: Bool {
        let marked = library.marks.allSatisfy { choice in draft.song.parts.contains { $0.mark == choice.id } }
        return marked && !title.trimmingCharacters(in: .whitespaces).isEmpty && !keeping
    }

    private var pinch: some Gesture {
        MagnifyGesture()
            .onChanged { pinch in
                let start = pinched ?? zoom
                pinched = start
                zoom = min(max(start * pinch.magnification, 1), Metrics.Sound.mostZoom)
            }
            .onEnded { _ in pinched = nil }
    }

    /// The stretch dragged over, played over and over; nil once it is let go, which ends it.
    private func hear(_ range: ClosedRange<Double>?, _ sound: URL) {
        if let range { library.player.loop(range, in: sound) } else { library.player.endLoop() }
    }

    private func mark(_ range: ClosedRange<Double>, as mark: String?) {
        Task {
            do {
                try await library.mark(range, as: mark)
            } catch {
                problem = error.localizedDescription
            }
        }
    }

    private func keep() {
        keeping = true
        Task {
            do {
                try await library.keep(title: title, artist: artist, picture: picture)
            } catch {
                problem = error.localizedDescription
            }
            keeping = false
        }
    }
}
