// The song library, in a window of its own, which the Window menu opens, and so does the form of
// a new session to choose its song: the songs in a sidebar, as Apple Music lists them (the
// cover, the title and the artist under it), the one selected beside it (HIG "Sidebars", "Split
// views"). A song is added with the button at the foot of the sidebar, as a session is, and
// changed in the editor, a sheet over the window (SongEditor), where a song added is first
// marked before it goes into the library; it is deleted as a session is: with its context
// menu, Edit ▸ Delete, the Delete key or the toolbar's button, then confirmed, as it cannot be
// undone (HIG "Alerts"). While the form asks for a song and one is selected, Choose at the foot
// of the window takes it; otherwise the window has no foot.
import SwiftUI
import UniformTypeIdentifiers

/// The window of the song library, once the program is ready.
struct LibraryWindow: View {
    static let id = "library"
    @Environment(Program.self) private var program

    var body: some View {
        if let library = program.library {
            LibraryView(library: library)
        } else {
            ContentUnavailableView(
                "AdaMus Is Not Ready", systemImage: "music.note.list",
                description: Text("The song library shows its songs while AdaMus runs."))
        }
    }
}

struct LibraryView: View {
    @Bindable var library: LibraryModel
    @State private var adding = false
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        NavigationSplitView(columnVisibility: .constant(.all)) {  // the sidebar is never hidden
            SongList(library: library) { adding = true }
                .toolbar(removing: .sidebarToggle)
                .navigationSplitViewColumnWidth(
                    min: Metrics.sidebar.min, ideal: Metrics.sidebar.ideal, max: Metrics.sidebar.max)
        } detail: {
            detail
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .safeAreaInset(edge: .bottom) {
                    if let choose = library.choose, let song = library.selected { chooseBar(song, choose) }
                }
        }
        // a toolbar with nothing in it is none, and the window lower without it: as high whether
        // anything is chosen or not
        .toolbar { ToolbarSpacer(.flexible) }
        .fileImporter(isPresented: $adding, allowedContentTypes: [.audio]) {
            if case .success(let url) = $0 { Task { await library.add(url) } }
        }
        .sheet(item: Binding(get: { library.draft }, set: { if $0 == nil { Task { await library.discard() } } })) {
            SongEditor(draft: $0, library: library)
        }
        .alert(
            "Delete “\(library.toDelete?.title ?? "")”?",
            isPresented: Binding(get: { library.toDelete != nil }, set: { if !$0 { library.toDelete = nil } }),
            presenting: library.toDelete
        ) { song in
            // no default: a Return pressed by chance deletes nothing (HIG "Alerts")
            Button("Delete", role: .destructive) { Task { await library.delete(song) } }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text(
                "The song is deleted from the library; the sessions that played it keep their own copy. You can’t undo this action."
            )
        }
        .alert(
            "The Request Could Not Be Done",
            isPresented: Binding(get: { library.problem != nil }, set: { if !$0 { library.problem = nil } })
        ) {
            Button("OK") {}
        } message: {
            Text(library.problem ?? "")
        }
        .onChange(of: library.selection) { library.player.stop() }  // what played was the song left
        .onDisappear {
            library.player.stop()
            library.choose = nil  // closed without a song chosen
        }
        .task { await library.load() }
    }

    @ViewBuilder private var detail: some View {
        if let song = library.selected {
            SongView(song: song, library: library)
                .id(song.id)
        } else if let problem = library.listProblem, library.songs.isEmpty {
            ContentUnavailableView {
                Label("The Songs Could Not Be Read", systemImage: "exclamationmark.triangle")
            } description: {
                Text(problem)
            } actions: {
                Button("Try Again") { Task { await library.load() } }
            }
        } else if library.songs.isEmpty && library.adding == nil {
            ContentUnavailableView {
                Label("No Songs", systemImage: "music.note.list")
            } description: {
                Text("A song you add appears here.")
            } actions: {
                Button("Add Song") { adding = true }
                    .disabled(!library.canAdd)
            }
        } else {
            ContentUnavailableView(
                "No Song Selected", systemImage: "music.note",
                description: Text("Choose a song in the sidebar."))
        }
    }

    /// Choose, which takes the song selected for the form of a new session.
    private func chooseBar(_ song: Song, _ choose: @escaping (Song.ID) -> Void) -> some View {
        HStack {
            Spacer()
            Button("Choose") {
                choose(song.id)
                library.choose = nil
                dismissWindow(id: LibraryWindow.id)
            }
            .keyboardShortcut(.defaultAction)
        }
        .padding(Metrics.Spacing.regular)
        .background(.bar)
    }
}

/// The songs, by title, and those still being analysed; the button that adds one at its foot.
struct SongList: View {
    @Bindable var library: LibraryModel
    let add: () -> Void

    var body: some View {
        List(selection: $library.selection) {
            if let problem = library.listProblem {
                Label(problem, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
            }
            ForEach(library.songs) { song in
                SongRow(song: song, cover: library.api.song(song).cover)
                    .tag(song.id)
            }
            if let name = library.adding {
                Label {
                    Text(name).lineLimit(1)
                } icon: {
                    ProgressView().controlSize(.small)
                }
                .foregroundStyle(.secondary)
                .help("The song is analysed")
            }
        }
        .contextMenu(forSelectionType: Song.ID.self) { chosen in
            Button("Delete…") { library.askToDelete(chosen.first) }
                .disabled(chosen.count != 1)
        }
        .onDeleteCommand { library.askToDelete(library.selection) }
        .safeAreaInset(edge: .bottom) {  // as New Session at the foot of the sessions
            Button("Add Song", systemImage: "plus", action: add)
                .disabled(!library.canAdd)
                .buttonStyle(.borderless)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(Metrics.Spacing.regular)
        }
    }
}

/// A song in the list, as Apple Music lists one: its cover, and its title over its artist.
struct SongRow: View {
    let song: Song
    let cover: URL?

    var body: some View {
        HStack(spacing: Metrics.Spacing.small) {
            Cover(picture: cover, side: Metrics.Sound.thumbnail)
            VStack(alignment: .leading, spacing: Metrics.Spacing.line) {
                Text(song.title).lineLimit(1)
                Text(song.artist ?? unknownArtist).font(.callout).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .help(song.title)
    }
}
