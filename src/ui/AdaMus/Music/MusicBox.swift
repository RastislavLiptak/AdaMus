// The music of a session on its overview: the cover of its style, played from the button on it,
// and beside it the style's name, the sound track of its sample (how loud along it, in the colour
// of the state each part is played in, what has played bright as a slider fills), its two forms
// under the parts of the track they are, and its other settings in a row, as the overview's facts.
// A click or a drag along the track puts the sample there (HIG "Sliders").
import SwiftUI

struct MusicBox: View {
    let music: Music
    let sample: URL
    let cover: URL
    /// The session's: it pauses the sample as the overview is left (SessionModel).
    let player: SamplePlayer

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Metrics.Spacing.regular) {
                Label {
                    Text("Music")
                } icon: {
                    Image(systemName: "circle.fill").imageScale(.small).foregroundStyle(Colour.pink.color)
                }
                .font(.headline)
                HStack(alignment: .top, spacing: Metrics.Spacing.page) {
                    coverWithButton
                    VStack(alignment: .leading, spacing: Metrics.Spacing.small) {
                        Text(music.title).font(.title).fontWeight(.bold)
                        SamplePosition(music: music, player: player)
                        HStack(alignment: .top, spacing: Metrics.Spacing.large) {
                            ForEach(music.parts, id: \.self) { part($0) }
                        }
                        Divider().padding(.vertical, Metrics.Spacing.tight)
                        // all in a row, as wide as the track
                        HStack(alignment: .top, spacing: Metrics.Spacing.regular) {
                            ForEach(music.facts, id: \.self) {
                                FactView(fact: $0).frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                }
            }
            .padding(Metrics.Spacing.small)
        }
    }

    private var coverWithButton: some View {
        AsyncImage(url: cover) { image in
            image.resizable().aspectRatio(contentMode: .fill)
        } placeholder: {
            Rectangle().fill(.quaternary)
        }
        .frame(width: Metrics.Sample.cover, height: Metrics.Sample.cover)
        .clipShape(.rect(cornerRadius: Metrics.cornerRadius))
        .shadow(radius: Metrics.Sample.coverShadow, y: Metrics.Spacing.line)
        .overlay {
            Button {
                if player.playing { player.pause() } else { player.play(sample) }
            } label: {
                Image(systemName: player.playing ? "pause.fill" : "play.fill")
                    .font(.title)
                    .frame(width: Metrics.Sample.playButton, height: Metrics.Sample.playButton)
            }
            .buttonStyle(.glass)  // the system's own: the whole round takes a press
            .buttonBorderShape(.circle)
            .accessibilityLabel(player.playing ? "Pause Sample" : "Play Sample")
            .help(player.playing ? "Pause the sample" : "Play a sample of the music")
        }
    }

    private func part(_ part: MusicPart) -> some View {
        VStack(alignment: .leading, spacing: Metrics.Spacing.tight) {
            Label {
                Text(part.title)
            } icon: {
                Image(systemName: "circle.fill").imageScale(.small).foregroundStyle(part.colour.color)
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
            Text("“\(part.prompt)”").font(.callout).fixedSize(horizontal: false, vertical: true)
            Text(part.sound).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Where the sample is: its sound track, and the time played and left under it. As the sample
/// plays, the player is asked where it is as each frame is drawn (SwiftUI's TimelineView, not
/// the timeline of a session), and only this is worked out again; paused, nothing is, and it
/// shows where the player stopped.
struct SamplePosition: View {
    let music: Music
    let player: SamplePlayer

    var body: some View {
        SwiftUI.TimelineView(.animation(paused: !player.playing)) { _ in
            let time = min(player.time, music.lengthS)
            VStack(alignment: .leading, spacing: Metrics.Spacing.small) {
                SoundTrack(music: music, shown: time) { player.seek(to: $0) }
                HStack {
                    Text(ValueFormatting.clock(time))
                    Spacer()
                    Text("−" + ValueFormatting.clock(music.lengthS - time))
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            }
        }
    }
}

/// How loud the sample is along it, a bar a slice, in the colour of the part each is in: the bars
/// faint, and what has played over them bright, up to a line where the sample is. Drawn once;
/// as the sample plays, only how much of the bright bars shows and where the line is change.
/// A click or a drag along it puts the sample there.
struct SoundTrack: View {
    let music: Music
    let shown: Double
    let seek: (Double) -> Void
    @State private var width: CGFloat = 0

    var body: some View {
        // scaled and moved as they are drawn, which lays nothing out again
        let share = min(max(shown / music.lengthS, 0), 1)
        ZStack(alignment: .leading) {
            SoundBars(music: music).equatable().opacity(Metrics.Sample.unplayed)
            SoundBars(music: music).equatable()
                .mask { Rectangle().scaleEffect(x: share, y: 1, anchor: .leading) }
            Rectangle()
                .fill(.primary)
                .frame(width: Metrics.Sample.playhead)
                .offset(x: width * share - Metrics.Sample.playhead / 2)
        }
        .frame(height: Metrics.Sample.track)
        .contentShape(.rect)
        .gesture(
            DragGesture(minimumDistance: 0).onChanged { drag in
                guard width > 0 else { return }
                seek(min(max(drag.location.x / width, 0), 1) * music.lengthS)
            }
        )
        .onGeometryChange(for: CGFloat.self) {
            $0.size.width
        } action: {
            width = $0
        }
        .accessibilityElement()
        .accessibilityLabel("Sample")
        .accessibilityValue("\(ValueFormatting.clock(shown)) of \(ValueFormatting.clock(music.lengthS))")
        .accessibilityAdjustableAction { direction in
            let step = direction == .increment ? Metrics.Sample.step : -Metrics.Sample.step
            seek(min(max(shown + step, 0), music.lengthS))
        }
    }
}

/// How loud the sample is along it, a bar a slice, in the colour of the part each is in.
struct SoundBars: View, Equatable {
    let music: Music

    var body: some View {
        Canvas { context, size in
            let slice = size.width / CGFloat(max(music.loudness.count, 1))
            for (index, level) in music.loudness.enumerated() {
                let at = (Double(index) + 0.5) / Double(music.loudness.count) * music.lengthS
                let height = max(Metrics.Sample.leastBar, level) * size.height
                let bar = CGRect(
                    x: CGFloat(index) * slice, y: (size.height - height) / 2,
                    width: slice * Metrics.Sample.barShare, height: height)
                let colour = (music.parts.last { $0.fromS <= at } ?? music.parts.first)?.colour.color ?? .secondary
                context.fill(Capsule().path(in: bar), with: .color(colour))
            }
        }
    }
}
