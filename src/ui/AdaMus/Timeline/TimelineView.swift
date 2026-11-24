// The timeline, as an editor of video shows one: the transport and a ruler on top, then the
// tracks in groups that fold away, each a name beside its lane of a fixed height; the lanes
// scroll up and down, and sideways together in one scroll view, drawn in tiles that scrolling only
// moves; the playhead runs across them. The toolbar zooms, and so does a pinch. A drag with
// Shift marks a part of the session, cut out of its data (MarkedPartSheet).
import SwiftUI

struct TimelineView: View {
    let model: TimelineModel
    /// The electrodes on the head: the name of an electrode's track opens where it is.
    let electrodes: ElectrodeSet?
    @State private var folded: Set<String> = []

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                Transport(model: model)
                    .frame(width: Metrics.trackNames)
                Ruler(model: model)
            }
            .frame(height: Metrics.ruler)
            Divider()
            let rows = TimelineRow.rows(of: model.timeline.groups, folded: folded)
            ScrollView(.vertical) {
                HStack(alignment: .top, spacing: 0) {
                    TrackNames(rows: rows, folded: $folded, electrodes: electrodes)
                        .frame(width: Metrics.trackNames)
                    LaneScroller(model: model, rows: rows)
                }
            }
        }
        .overlay(alignment: .topLeading) {
            MarkedPartView(model: model)
                .padding(.leading, Metrics.trackNames)
                .allowsHitTesting(false)
        }
        .overlay(alignment: .topLeading) {
            Playhead(model: model)
                .padding(.leading, Metrics.trackNames)
                .allowsHitTesting(false)
        }
    }
}

/// The lanes of the whole session, which the scroll view scrolls across: a view of its own, so
/// that a scroll (by the operator, a zoom, or the playhead turning a page) changes only it.
struct LaneScroller: View {
    @Bindable var model: TimelineModel
    let rows: [TimelineRow]
    @State private var pinchStart: Double?  // the seconds in view when the pinch began

    var body: some View {
        ScrollView(.horizontal) {
            if model.pointsPerSecond > 0 {
                Lanes(
                    rows: rows, pyramids: model.pyramids, start: model.timeline.startS, duration: model.span,
                    pointsPerSecond: model.pointsPerSecond
                )
                .equatable()
                .gesture(  // a click or a drag moves the playhead there, as on the ruler
                    DragGesture(minimumDistance: 0).onChanged { drag in model.seek(toLanesX: drag.location.x) }
                )
                .highPriorityGesture(  // with Shift held, a drag marks a part instead
                    DragGesture(minimumDistance: 0).modifiers(.shift)
                        .onChanged { drag in model.mark(fromLanesX: drag.startLocation.x, toLanesX: drag.location.x) }
                        .onEnded { _ in model.endMarking() }
                )
            }
        }
        .scrollPosition($model.position)
        .onScrollGeometryChange(for: Double.self) {
            $0.contentOffset.x
        } action: { _, offset in
            model.scrolled(toOffset: offset)
        }
        .onGeometryChange(for: Double.self) {
            $0.size.width
        } action: {
            model.plotWidthChanged(to: $0)
        }
        .gesture(
            MagnifyGesture()
                .onChanged { pinch in
                    let start = pinchStart ?? model.visibleSeconds
                    pinchStart = start
                    let under = model.scrollX + pinch.startAnchor.x * model.visibleSeconds  // the fingers
                    model.zoom(by: model.visibleSeconds / (start / pinch.magnification), around: under)
                }
                .onEnded { _ in pinchStart = nil }
        )
        .frame(height: rows.reduce(0) { $0 + $1.height })
    }
}

/// A row of the timeline: the title of a group, or a track of an unfolded group, as high as the
/// window gives it.
struct TimelineRow: Identifiable, Equatable {
    enum Kind {
        case group(TrackGroup)
        case track(Track, colour: Colour)
    }

    let kind: Kind
    let height: CGFloat

    /// The same row, as a drawing depends on it: which it is, and how high.
    nonisolated static func == (one: TimelineRow, other: TimelineRow) -> Bool {
        one.id == other.id && one.height == other.height
    }

    var id: String {
        switch kind {
        case .group(let group): "group:\(group.title)"
        case .track(let track, _): "track:\(track.id)"
        }
    }

    /// Every group, and the tracks of those not folded, each as high as its kind of row is.
    static func rows(of groups: [TrackGroup], folded: Set<String>) -> [TimelineRow] {
        let kinds: [Kind] = groups.flatMap { group in
            [.group(group)]
                + (folded.contains(group.title) ? [] : group.tracks.map { .track($0, colour: group.colour) })
        }
        return kinds.map { TimelineRow(kind: $0, height: height(of: $0)) }
    }

    private static func height(of kind: Kind) -> CGFloat {
        switch kind {
        case .group: Metrics.Row.group
        case .track(.curve(let curve), _): Metrics.Row.curve + (curve.bands.isEmpty ? 0 : Metrics.Row.strip)
        case .track(.signal, _), .track(.audio, _): Metrics.Row.signal
        }
    }
}

/// What each row is: the title of a group, which folds it, or the name of a track and what it is
/// read by; the name of an electrode on the head opens where it is.
struct TrackNames: View {
    let rows: [TimelineRow]
    @Binding var folded: Set<String>
    let electrodes: ElectrodeSet?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(rows) { row in
                Group {
                    switch row.kind {
                    case .group(let group):
                        let isFolded = folded.contains(group.title)
                        Button {
                            folded.formSymmetricDifference([group.title])
                        } label: {
                            HStack(spacing: Metrics.Spacing.symbol) {
                                Image(systemName: "chevron.right")
                                    .rotationEffect(.degrees(isFolded ? 0 : 90))
                                    .foregroundStyle(.secondary)
                                Image(systemName: "circle.fill").foregroundStyle(group.colour.color).imageScale(.small)
                                Text(group.title).font(.headline)
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(.isHeader)
                        .accessibilityValue(isFolded ? "Collapsed" : "Expanded")
                    case .track(let track, _):
                        VStack(alignment: .leading, spacing: Metrics.Spacing.line) {
                            if case .signal = track, let electrodes, let electrode = electrodes.named(track.title) {
                                ElectrodeTrackName(set: electrodes, electrode: electrode)
                            } else {
                                Text(track.title)
                            }
                            if let caption = track.caption {
                                Text(caption).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .padding(.horizontal)
                .frame(maxWidth: .infinity, minHeight: row.height, maxHeight: row.height, alignment: .leading)
            }
        }
    }
}

/// Play and pause, and where the playhead is; a sound that cannot be played says why.
struct Transport: View {
    let model: TimelineModel

    var body: some View {
        HStack(spacing: Metrics.Spacing.symbol) {
            if let problem = model.playbackProblem {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
                    .help(problem)
                    .accessibilityLabel(problem)
            } else {
                Button(model.playing ? "Pause" : "Play", systemImage: model.playing ? "pause.fill" : "play.fill") {
                    model.togglePlayback()
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
                .disabled(!model.canPlay)
                .help(model.playing ? "Pause" : "Play")
            }
            Text(ValueFormatting.clock(model.playhead, tenths: true))
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The time across the top; a click or a drag on it, or on the lanes, moves the playhead there.
struct Ruler: View {
    let model: TimelineModel

    var body: some View {
        let scale = model.scale
        Canvas { context, size in
            guard scale.pointsPerSecond > 0 else { return }
            let step = TickSpacing.seconds(forPointsPerSecond: scale.pointsPerSecond)
            var tick = (scale.leading / step).rounded(.down) * step
            while scale.x(of: tick) <= size.width {
                let x = scale.x(of: tick)
                if x >= 0 {
                    context.fill(
                        Path(
                            CGRect(x: x, y: size.height - Metrics.tick, width: Metrics.hairline, height: Metrics.tick)),
                        with: .style(.secondary))
                    context.draw(
                        Text(ValueFormatting.clock(tick)).font(.caption).monospacedDigit().foregroundStyle(.secondary),
                        at: CGPoint(x: x + Metrics.tickLabel.dx, y: size.height / 2 + Metrics.tickLabel.dy),
                        anchor: .leading)
                }
                tick += step
            }
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { drag in model.seek(to: model.scale.time(at: drag.location.x)) }
        )
    }
}

/// The line of the playhead across the ruler and every lane, drawn over them, so that they are
/// not drawn again as it moves.
struct Playhead: View {
    let model: TimelineModel

    var body: some View {
        let x = model.scale.x(of: model.playhead)
        if model.pointsPerSecond > 0, x >= 0, x <= model.plotWidth {
            Rectangle()
                .fill(Color.accentColor)
                .frame(width: Metrics.playhead)
                // whole also at the edges of the view, as at the start of the session
                .offset(x: min(max(x - Metrics.playhead / 2, 0), model.plotWidth - Metrics.playhead))
                .accessibilityHidden(true)
        }
    }
}

/// The part marked, tinted across the ruler and every lane.
struct MarkedPartView: View {
    let model: TimelineModel

    var body: some View {
        if let part = model.marked, model.pointsPerSecond > 0 {
            let start = max(0, model.scale.x(of: part.startS))
            let end = min(model.plotWidth, model.scale.x(of: part.endS))
            if end > start {
                Rectangle()
                    .fill(.tint.quaternary)
                    .frame(width: end - start)
                    .offset(x: start)
                    .accessibilityHidden(true)
            }
        }
    }
}
