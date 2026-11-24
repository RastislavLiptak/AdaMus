// The electrodes of a session on the head, in two places. On the overview, a box: the head, which
// a drag turns; beside it, until one is chosen, what to do, then the name of the chosen electrode,
// what it records (scrolling when it is long) and a line from the name to its dot. On the
// timeline, the name of an electrode's track opens a popover: the same head, still, turned to
// that electrode, its name and what it records under it (HIG "Popovers"). Both draw the head the
// same way (HeadRealityView); what is not in one of them is not made there at all.
import RealityKit
import SwiftUI
import simd

/// The head drawn by RealityKit: made once, painted for the appearance, and, where it moves,
/// advanced every frame.
struct HeadRealityView: View {
    let set: ElectrodeSet
    let dotRadius: Float
    /// What the head shows once it is made.
    let made: (HeadScene) -> Void
    /// What a frame of so many seconds does to it; nil where it stays still.
    var frame: ((HeadScene, Float) -> Void)?
    @Environment(\.colorScheme) private var scheme
    @State private var scene: HeadScene?
    @State private var frames: EventSubscription?

    var body: some View {
        RealityView { content in
            guard let made = try? HeadScene(set: set, dotRadius: dotRadius) else { return }
            made.paint(for: scheme)
            self.made(made)
            content.add(made.root)
            content.add(made.camera)
            if let frame {
                frames = content.subscribe(to: SceneEvents.Update.self) { event in
                    frame(made, Float(event.deltaTime))
                }
            }
            scene = made
        } update: { _ in
            scene?.paint(for: scheme)
        }
    }
}

// MARK: - The overview

struct ElectrodesBox: View {
    @State private var turner: HeadTurner
    @State private var head: CGRect = .zero
    @State private var name: CGRect = .zero
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    nonisolated private static let space = "electrodes"

    init(set: ElectrodeSet) {
        _turner = State(initialValue: HeadTurner(set: set))
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: Metrics.Spacing.small) {
                Label {
                    Text("Electrodes")
                } icon: {
                    Image(systemName: "circle.fill").imageScale(.small).foregroundStyle(Colour.teal.color)
                }
                .font(.headline)
                HStack(spacing: Metrics.Head.textGap) {
                    TurningHead(turner: turner)
                        .onGeometryChange(for: CGRect.self) {
                            $0.frame(in: .named(Self.space))
                        } action: {
                            head = $0
                        }
                    // there while nothing is shown too, so that the head keeps its place
                    Color.clear.overlay(alignment: .leading) { details }
                }
                .frame(height: Metrics.Head.box)
                .overlay { line }
                .coordinateSpace(.named(Self.space))
            }
            .padding(Metrics.Spacing.small)
        }
    }

    @ViewBuilder private var details: some View {
        if let shown = turner.shown, let electrode = turner.set.named(shown) {
            VStack(alignment: .leading, spacing: Metrics.Spacing.tight) {
                Text(electrode.name)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .onGeometryChange(for: CGRect.self) {
                        $0.frame(in: .named(Self.space))
                    } action: {
                        name = $0
                    }
                AboutText(text: electrode.about)
            }
            .id(shown)
            .transition(.opacity.combined(with: .offset(x: Metrics.Head.appearShift)))
            .animation(reduceMotion ? nil : .easeOut(duration: Metrics.Head.appear), value: shown)
        } else if turner.chosen == nil {
            Text("Click an electrode to see what it records.")
                .foregroundStyle(.secondary)
        }
    }

    /// From the name of the shown electrode to its dot, while the dot faces the viewer.
    @ViewBuilder private var line: some View {
        if turner.shown != nil, let point = turner.shownPoint, name != .zero {
            let dot = CGPoint(x: head.minX + point.x, y: head.minY + point.y)
            let end = CGPoint(x: name.minX - Metrics.Head.lineGap, y: name.midY)
            let length = max(hypot(end.x - dot.x, end.y - dot.y), 1)
            let start = CGPoint(
                x: dot.x + (end.x - dot.x) / length * Metrics.Head.lineGap,
                y: dot.y + (end.y - dot.y) / length * Metrics.Head.lineGap)  // from the edge of the ring
            Path { path in
                path.move(to: start)
                path.addLine(to: end)
            }
            .stroke(.secondary, style: StrokeStyle(lineWidth: Metrics.Head.line, lineCap: .round))
            .allowsHitTesting(false)
            .accessibilityHidden(true)
        }
    }
}

/// The head of the overview, turned by a drag, an electrode chosen by a click.
struct TurningHead: View {
    let turner: HeadTurner
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        HeadRealityView(set: turner.set, dotRadius: Metrics.Head.dot) { scene in
            turner.step(by: 0, in: scene)
        } frame: { scene, seconds in
            turner.step(by: seconds, in: scene)
        }
        .onGeometryChange(for: CGSize.self) {
            $0.size
        } action: {
            turner.size = $0
        }
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { drag in
                    if hypot(drag.translation.width, drag.translation.height) > Metrics.Head.click {
                        turner.drag(to: drag.translation)
                    }
                }
                .onEnded { drag in turner.release(at: drag.location, after: drag.translation) }
        )
        .onContinuousHover { phase in
            switch phase {
            case .active(let location): turner.hover(at: location)
            case .ended: turner.hover(at: nil)
            }
        }
        .pointerStyle(turner.hovered == nil ? .grabIdle : .link)
        .onChange(of: reduceMotion, initial: true) { turner.reduceMotion = reduceMotion }
        .overlay { DotsForVoiceOver(turner: turner) }
    }
}

/// For VoiceOver, a button on each dot that faces the viewer, where the dot is; the pointer goes
/// through to the head under it. (Not accessibilityChildren on the head: with its size measured,
/// it made SwiftUI build the RealityView anew, and the new one drew nothing.)
struct DotsForVoiceOver: View {
    let turner: HeadTurner

    var body: some View {
        ZStack {
            ForEach(turner.set.placed) { electrode in
                if let at = turner.point(of: electrode.name) {
                    Color.clear
                        .frame(width: 2 * Metrics.Head.reach, height: 2 * Metrics.Head.reach)
                        .position(at)
                        .accessibilityElement()
                        .accessibilityLabel(electrode.name)
                        .accessibilityAddTraits(.isButton)
                        .accessibilityAction { turner.choose(electrode.name) }
                }
            }
        }
        .allowsHitTesting(false)
    }
}

/// What an electrode records: as it is where it fits, scrolling where it does not, its lower
/// edge fading while more of it lies below.
struct AboutText: View {
    let text: String
    var limit: CGFloat?

    var body: some View {
        let shown = Text(text).foregroundStyle(.secondary).textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        ViewThatFits(in: .vertical) {
            shown.fixedSize(horizontal: false, vertical: true)
            ScrollView { shown }
                .scrollEdgeEffectStyle(.soft, for: .bottom)
        }
        .frame(maxHeight: limit)
    }
}

// MARK: - The timeline

/// The name of an electrode's track, by a dot of its colour: it opens the popover of the electrode.
/// Under the pointer, and while its popover is open, the name takes the accent colour.
struct ElectrodeTrackName: View {
    let set: ElectrodeSet
    let electrode: Electrode
    @State private var open = false
    @State private var hovered = false

    var body: some View {
        Button {
            open.toggle()
        } label: {
            Label {
                Text(electrode.name)
                    .foregroundStyle(hovered || open ? AnyShapeStyle(.tint) : AnyShapeStyle(.primary))
            } icon: {
                Image(systemName: "circle.fill").imageScale(.small)
                    .foregroundStyle(set.colours[electrode.name] ?? .gray)
            }
        }
        .buttonStyle(.plain)
        .pointerStyle(.link)
        .onHover { hovered = $0 }
        .help("Where \(electrode.name) is, and what it records")
        .popover(isPresented: $open, arrowEdge: .trailing) {
            ElectrodePopover(set: set, electrode: electrode)
        }
    }
}

/// The head, still, turned to one electrode with its ring, the others dimmed; its name and what
/// it records under it.
struct ElectrodePopover: View {
    let set: ElectrodeSet
    let electrode: Electrode

    var body: some View {
        VStack(alignment: .leading, spacing: Metrics.Spacing.small) {
            HeadRealityView(set: set, dotRadius: Metrics.Head.popoverDot) { scene in
                if let point = HeadSpace.point(of: electrode) { scene.turn(HeadTurn.facing(simd_normalize(point))) }
                scene.show(chosen: electrode.name)
                for other in set.placed where other.name != electrode.name {
                    scene.light(other.name, Float(Palette.Head.dimmed))
                }
            }
            .frame(height: Metrics.Head.popover.height)
            .accessibilityLabel("A head, \(electrode.name) on it")
            Label {
                Text(electrode.name)
            } icon: {
                Image(systemName: "circle.fill").imageScale(.small)
                    .foregroundStyle(set.colours[electrode.name] ?? .gray)
            }
            .font(.headline)
            AboutText(text: electrode.about, limit: Metrics.Head.popoverText)
        }
        .padding()
        .frame(width: Metrics.Head.popover.width)
    }
}
