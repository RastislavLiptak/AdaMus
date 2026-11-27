// The source of the music of a session on its overview: its song as the session kept it when its
// music began, whatever the library holds since (its sound, its parts as they were marked, its
// title, artist and picture), shown and played as the library shows a song: the cover, played
// from the button on it, the title and the artist beside it, and the sound track in the colours
// of its parts with what each colour means. What the session played is on its timeline.
import SwiftUI

struct MusicSourceBox: View {
    let song: Song
    let sound: URL
    let cover: URL?
    /// The session's: it pauses the song as the overview is left (SessionModel).
    let player: SoundPlayer

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Metrics.Spacing.regular) {
                Label {
                    Text("Music Source")
                } icon: {
                    Image(systemName: "circle.fill").imageScale(.small).foregroundStyle(Colour.pink.color)
                }
                .font(.headline)
                HStack(alignment: .top, spacing: Metrics.Spacing.page) {
                    PlayableCover(picture: cover, sound: sound, player: player, name: "song")
                    VStack(alignment: .leading, spacing: Metrics.Spacing.small) {
                        SongHeading(title: song.title, artist: song.artist)
                        SoundPosition(shape: song, sound: sound, player: player, name: "song")
                        PartsLegend(shape: song)
                    }
                }
            }
            .padding(Metrics.Spacing.small)
        }
    }
}
