// What a song and the music of a session are shown by alike: the cover, the picture of the song
// or a note where it has none, as Apple Music shows a song, with the button that plays it; the
// sound track (how loud along it, in the colour of the part each slice is in, what has played
// bright as a slider fills) with the time played and left under it, and what each colour means.
// A click or a drag along the track puts the sound there (HIG "Sliders"); where its parts can be
// marked, a drag selects a stretch, as text is, and the stretch plays over and over from its
// start until the drag is let go, apart from the sound itself, which stops and stays where it
// was; a drag may begin a little beside the track, so that its very start and end are easy to
// reach; its end clings to the edge of a part it comes near, so that parts meet without a gap;
// a menu over it, as the edit menu of iOS, marks it a state or removes its mark (HIG
// "Popovers").
import SwiftUI

/// What a sound track is drawn from: how loud along it, and its parts in their colours.
protocol SoundShape: Hashable, Sendable {
    var loudness: [Double] { get }
    var lengthS: Double { get }
    var parts: [SongPart] { get }
}

extension SoundShape {
    /// The part at ``seconds``, if one is there.
    func part(at seconds: Double) -> SongPart? {
        parts.first { $0.startS <= seconds && seconds < $0.endS }
    }

    /// ``time``, or the edge of a part or of the sound it is within ``reach`` seconds of: a
    /// stretch dragged up to a part meets it, with no gap and no overlap.
    func clinging(_ time: Double, within reach: Double) -> Double {
        let edges = [0, lengthS] + parts.flatMap { part -> [Double] in [part.startS, part.endS] }
        let nearest = edges.min { abs($0 - time) < abs($1 - time) } ?? time
        return abs(nearest - time) <= reach ? nearest : time
    }

    /// How loud the sound is at the share ``share`` of its length: between two slices, as far
    /// from each as it lies.
    func loudness(at share: Double) -> Double {
        guard let last = loudness.indices.last else { return 0 }
        let place = min(max(share * Double(loudness.count) - 0.5, 0), Double(last))
        let below = Int(place)
        let above = min(below + 1, last)
        return loudness[below] + (loudness[above] - loudness[below]) * (place - Double(below))
    }

    /// Whether some of the sound is in no part.
    var hasUnmarked: Bool {
        parts.reduce(0) { $0 + $1.endS - $1.startS } < lengthS
    }

    /// Each kind of part once, in the order they first come.
    var kinds: [SongPart] {
        var seen = Set<String>()
        return parts.filter { seen.insert($0.mark).inserted }
    }
}

/// What a stretch of a track can be marked, and what marks it: the id of a choice, or nil, which
/// removes the mark; and what plays the stretch selected as it is dragged, nil once let go.
struct Marking {
    let choices: [MarkChoice]
    let mark: (ClosedRange<Double>, String?) -> Void
    let hear: (ClosedRange<Double>?) -> Void
}

/// What a song whose artist is not known is said to be by, as Apple Music says it.
let unknownArtist = "Unknown Artist"

/// The title of a song, and its artist under it.
struct SongHeading: View {
    let title: String
    let artist: String?

    var body: some View {
        VStack(alignment: .leading, spacing: Metrics.Spacing.line) {
            Text(title).font(.title).fontWeight(.bold)
            Text(artist ?? unknownArtist).font(.title3).foregroundStyle(.secondary)
        }
    }
}

/// The cover with a round button on its middle that plays the sound or pauses it.
struct PlayableCover: View {
    let picture: URL?
    let sound: URL
    let player: SoundPlayer
    let name: String  // of what plays, in the button's label and help

    var body: some View {
        Cover(picture: picture, side: Metrics.Sound.cover)
            .shadow(radius: Metrics.Sound.coverShadow, y: Metrics.Spacing.line)
            .overlay {
                let playing = player.plays(sound)
                Button {
                    if playing { player.pause() } else { player.play(sound) }
                } label: {
                    Image(systemName: playing ? "pause.fill" : "play.fill")
                        .font(.title)
                        .frame(width: Metrics.Sound.playButton, height: Metrics.Sound.playButton)
                }
                .buttonStyle(.glass)  // the system's own: the whole round takes a press
                .buttonBorderShape(.circle)
                .accessibilityLabel(playing ? "Pause \(name.capitalized)" : "Play \(name.capitalized)")
                .help(playing ? "Pause the \(name)" : "Play the \(name)")
            }
    }
}

/// The picture of the song, ``side`` wide and high; where it has none, or it cannot be read, a
/// note, as Apple Music shows a song without one.
struct Cover: View {
    let picture: URL?
    let side: CGFloat

    var body: some View {
        Group {
            if let picture {
                AsyncImage(url: picture) { phase in
                    switch phase {
                    case .success(let image): image.resizable().aspectRatio(contentMode: .fill)
                    case .empty: Rectangle().fill(.quaternary)
                    case .failure: note
                    @unknown default: note
                    }
                }
            } else {
                note
            }
        }
        .frame(width: side, height: side)
        .clipShape(.rect(cornerRadius: side * Metrics.Sound.coverRounding))
    }

    private var note: some View {
        Rectangle()
            .fill(.quaternary)
            .overlay {
                Image(systemName: "music.note")
                    .font(.system(size: side * Metrics.Sound.noteShare))
                    .foregroundStyle(.secondary)
            }
            .accessibilityHidden(true)
    }
}

/// Where the sound is: its sound track, and the time played and left under it. As it plays, the
/// player is asked where it is as each frame is drawn (SwiftUI's TimelineView, not the timeline
/// of a session), and only this is worked out again; paused, nothing is, and it shows where the
/// player stopped. Zoomed in, the track is wider than the view and scrolls; as the sound plays
/// out of what is in view, the page of it where it plays comes into view, as an editor turns it.
struct SoundPosition<Shape: SoundShape>: View {
    let shape: Shape
    let sound: URL
    let player: SoundPlayer
    let name: String  // of what plays, for VoiceOver
    var marking: Marking?
    /// How many times as wide as the view the track is.
    var zoom = 1.0
    var height = Metrics.Sound.track
    @State private var position = ScrollPosition(edge: .leading)
    @State private var visible: CGFloat = 0
    @State private var scrolled: CGFloat = 0

    var body: some View {
        SwiftUI.TimelineView(.animation(paused: !player.plays(sound) && !player.loops(sound))) { _ in
            let time = min(player.time(of: sound), shape.lengthS)
            VStack(alignment: .leading, spacing: Metrics.Spacing.small) {
                ScrollView(.horizontal) {
                    SoundTrack(
                        shape: shape, shown: time, heard: player.loopTime(of: sound), name: name, marking: marking
                    ) {
                        player.seek(to: $0, in: sound)
                    }
                    .frame(width: visible * zoom + 2 * Metrics.Sound.grab, height: height)
                }
                .padding(.horizontal, -Metrics.Sound.grab)  // its margins, where a drag may begin
                .scrollPosition($position)
                .scrollDisabled(zoom <= 1)
                .onScrollGeometryChange(for: CGFloat.self) {
                    $0.contentOffset.x
                } action: {
                    scrolled = $1
                }
                .onChange(of: page(at: time)) { _, page in
                    let at = time / shape.lengthS * visible * zoom  // from the start of the track
                    if player.plays(sound), at < scrolled || at > scrolled + visible {
                        position.scrollTo(x: CGFloat(page) * visible)
                    }
                }
                HStack {
                    Text(ValueFormatting.clock(time))
                    Spacer()
                    Text("−" + ValueFormatting.clock(shape.lengthS - time))
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            }
        }
        .onGeometryChange(for: CGFloat.self) {
            $0.size.width
        } action: {
            visible = $0
        }
        .onChange(of: zoom) { before, now in  // what was in the middle of the view stays there
            let middle = (scrolled + visible / 2) / (visible * before)
            position.scrollTo(x: max(0, middle * visible * now - visible / 2))
        }
    }

    /// Which width of the view the sound is in at ``time``, from the start of the track.
    private func page(at time: Double) -> Int {
        shape.lengthS > 0 ? Int(time / shape.lengthS * zoom) : 0
    }
}

/// How loud the sound is along it, in the colours of its parts: the bars faint, and what has
/// played over them bright, up to a line where the sound is. Drawn once; as the sound plays,
/// only how much of the bright bars shows and where the line is change, and so does how much of
/// a stretch heard over and over shows stronger. A click or a drag along it puts the sound there;
/// with a ``marking``, a drag selects the stretch it goes over, heard as it is dragged, and the
/// menu over it marks it.
struct SoundTrack<Shape: SoundShape>: View {
    let shape: Shape
    let shown: Double
    /// Where the stretch selected is heard, while it plays over and over: what of it has been
    /// heard shows stronger, as it fills, apart from the line of the sound, which stays put.
    var heard: Double?
    let name: String
    var marking: Marking?
    let seek: (Double) -> Void
    @State private var width: CGFloat = 0
    /// The stretch a drag goes over while it goes on.
    @State private var dragged: ClosedRange<Double>?
    /// The stretch selected once the drag ended, which the menu marks.
    @State private var selection: ClosedRange<Double>?

    var body: some View {
        // scaled and moved as they are drawn, which lays nothing out again
        let share = min(max(shown / shape.lengthS, 0), 1)
        ZStack(alignment: .leading) {
            SoundBars(shape: shape).equatable().opacity(Metrics.Sound.unplayed)
            SoundBars(shape: shape).equatable()
                .mask { Rectangle().scaleEffect(x: share, y: 1, anchor: .leading) }
            if let selected = dragged ?? selection, width > 0 {
                // laid out where it is, not moved there: the menu points at where it is laid out
                Rectangle()
                    .fill(.selection.opacity(Metrics.Sound.selected))
                    .frame(width: width * (selected.upperBound - selected.lowerBound) / shape.lengthS)
                    .padding(.leading, width * selected.lowerBound / shape.lengthS)
                    .popover(
                        isPresented: Binding(
                            get: { selection != nil && dragged == nil }, set: { if !$0 { selection = nil } }),
                        arrowEdge: .top
                    ) {
                        if let marking, let selection { menu(marking, selection) }
                    }
                if let heard {  // what of it has been heard this time round, the stronger
                    Rectangle()
                        .fill(.selection.opacity(Metrics.Sound.selected))
                        .frame(width: width * max(heard - selected.lowerBound, 0) / shape.lengthS)
                        .padding(.leading, width * selected.lowerBound / shape.lengthS)
                }
            }
            Rectangle()
                .fill(.primary)
                .frame(width: Metrics.Sound.playhead)
                .offset(x: width * share - Metrics.Sound.playhead / 2)
        }
        .onGeometryChange(for: CGFloat.self) {
            $0.size.width
        } action: {
            width = $0
        }
        .padding(.horizontal, Metrics.Sound.grab)
        .contentShape(.rect)
        .gesture(seeking, isEnabled: marking == nil)
        .gesture(selecting, isEnabled: marking != nil)
        .accessibilityElement()
        .accessibilityLabel("\(name.capitalized) Track")
        .accessibilityValue("\(ValueFormatting.clock(shown)) of \(ValueFormatting.clock(shape.lengthS))")
        .accessibilityAdjustableAction { direction in
            let step = direction == .increment ? Metrics.Sound.step : -Metrics.Sound.step
            seek(min(max(shown + step, 0), shape.lengthS))
        }
    }

    private var seeking: some Gesture {
        DragGesture(minimumDistance: 0).onChanged { seek(seconds(at: $0.location.x)) }
    }

    /// A drag selects from where it began, and is heard until it is let go; a press let go where
    /// it began puts the sound there and selects nothing.
    private var selecting: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { drag in
                guard abs(drag.translation.width) >= Metrics.Sound.leastDrag else { return }
                let from = seconds(at: drag.startLocation.x)
                let to = shape.clinging(
                    seconds(at: drag.location.x), within: Metrics.Sound.magnet / max(width, 1) * shape.lengthS)
                dragged = min(from, to)...max(from, to)
                marking?.hear(dragged)
            }
            .onEnded { drag in
                selection = dragged
                if dragged == nil { seek(seconds(at: drag.location.x)) } else { marking?.hear(nil) }
                dragged = nil
            }
    }

    /// A button for each state, and one that removes the mark where a part is marked.
    private func menu(_ marking: Marking, _ range: ClosedRange<Double>) -> some View {
        HStack(spacing: Metrics.Spacing.small) {
            ForEach(marking.choices) { choice in
                Button {
                    selection = nil
                    marking.mark(range, choice.id)
                } label: {
                    Label {
                        Text(choice.title)
                    } icon: {
                        Image(systemName: "circle.fill").imageScale(.small).foregroundStyle(choice.colour.color)
                    }
                    .modifier(MenuTarget())
                }
                .help("Mark the part selected \(choice.title.lowercased())")
            }
            if shape.parts.contains(where: { $0.startS < range.upperBound && range.lowerBound < $0.endS }) {
                Divider().frame(height: Metrics.Spacing.page)
                Button(role: .destructive) {
                    selection = nil
                    marking.mark(range, nil)
                } label: {
                    Text("Remove Mark").modifier(MenuTarget())
                }
                .help("Remove the mark of the part selected")
            }
        }
        .buttonStyle(.borderless)
        .focusEffectDisabled()  // nothing is chosen until a click chooses it
        .padding(Metrics.Spacing.small)
    }

    /// The time at ``x`` of the track with its margins: either margin, its end.
    private func seconds(at x: CGFloat) -> Double {
        width > 0 ? min(max((x - Metrics.Sound.grab) / width, 0), 1) * shape.lengthS : 0
    }
}

/// How loud the sound is along it, in bars all as wide and as far apart, as many as fit: a wider
/// track has more of them, not wider ones. Each is as loud as the sound where it lies, in the
/// colour of the part it is in, grey where none is.
struct SoundBars<Shape: SoundShape>: View, Equatable {
    let shape: Shape

    var body: some View {
        Canvas { context, size in
            let pitch = Metrics.Sound.bar + Metrics.Sound.barGap
            let count = Int((size.width + Metrics.Sound.barGap) / pitch)
            for index in 0..<count {
                let share = (Double(index) + 0.5) / Double(count)
                let height = max(Metrics.Sound.leastBar, shape.loudness(at: share)) * size.height
                let bar = CGRect(
                    x: CGFloat(index) * pitch, y: (size.height - height) / 2, width: Metrics.Sound.bar,
                    height: height)
                let colour = shape.part(at: share * shape.lengthS)?.colour.color ?? .secondary
                context.fill(Capsule().path(in: bar), with: .color(colour))
            }
        }
    }
}

/// What a button of the menu over a stretch is clicked by: all of it, as high as a click needs
/// (HIG "Accessibility": 28 by 28 points, the default size of a control on the Mac), not only
/// its text.
private struct MenuTarget: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(.horizontal, Metrics.Spacing.small)
            .frame(minHeight: Metrics.Sound.hitRegion)
            .contentShape(.rect)
    }
}

/// What each colour of the track means: each kind of its parts once, and, with ``unmarked``, what
/// is grey, where something is.
struct PartsLegend<Shape: SoundShape>: View {
    let shape: Shape
    var unmarked: String?

    var body: some View {
        HStack(spacing: Metrics.Spacing.large) {
            ForEach(shape.kinds, id: \.mark) { entry($0.title, $0.colour.color) }
            if let unmarked, shape.hasUnmarked { entry(unmarked, .secondary) }
        }
        .labelStyle(.titleAndIcon)  // as it looks anywhere, also in a form, which spaces it apart
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private func entry(_ title: String, _ colour: Color) -> some View {
        Label {
            Text(title)
        } icon: {
            Image(systemName: "circle.fill").imageScale(.small).foregroundStyle(colour)
        }
    }
}
