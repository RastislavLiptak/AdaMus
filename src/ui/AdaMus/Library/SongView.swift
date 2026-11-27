// A song of the library, beside the list: its cover, played from the button on it, its title and
// artist, and its sound track in the colours of its parts, as the overview shows the music of a
// session. Its toolbar deletes it, as that of a session does, and opens the editor, where
// everything of it is changed (SongEditor).
import SwiftUI

struct SongView: View {
    let song: Song
    let library: LibraryModel

    var body: some View {
        let (sound, cover) = library.api.song(song)
        ScrollView {
            HStack(alignment: .top, spacing: Metrics.Spacing.page) {
                PlayableCover(picture: cover, sound: sound, player: library.player, name: "song")
                VStack(alignment: .leading, spacing: Metrics.Spacing.small) {
                    SongHeading(title: song.title, artist: song.artist)
                    SoundPosition(shape: song, sound: sound, player: library.player, name: "song")
                    PartsLegend(shape: song, unmarked: library.unmarked)
                }
            }
            .padding(Metrics.Spacing.page)
        }
        .toolbar {
            ToolbarItem {
                Button("Delete…", systemImage: "trash") { library.askToDelete(song.id) }
                    .foregroundStyle(.red)
                    .help("Delete This Song")
            }
            ToolbarSpacer(.fixed)
            ToolbarItem {
                Button("Edit…", systemImage: "pencil") { Task { await library.edit(song.id) } }
                    .help("Edit This Song")
            }
        }
    }
}
