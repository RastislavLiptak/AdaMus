// What the head of the overview does: a drag turns it; a click on an electrode chooses it, and
// the head turns it to the viewer on a spring, from wherever it is, so that another click or a
// drag takes over at any moment (with Reduce Motion it is turned at once); its text comes as the
// head nearly faces it. The dot under the pointer lights up, the others dim while one is chosen,
// each easing to how lit it should be. Advanced once a frame by the scene it turns.
import Observation
import SwiftUI
import simd

@MainActor @Observable
final class HeadTurner {
    let set: ElectrodeSet
    private(set) var chosen: String?
    /// The chosen electrode, once the head nearly faces it: its name and text are shown.
    private(set) var shown: String?
    private(set) var hovered: String?
    /// How far the head is turned, as it is drawn now.
    private(set) var turn = HeadTurn.opening
    /// The size of the view the head is drawn in.
    var size: CGSize = .zero
    @ObservationIgnored var reduceMotion = false

    @ObservationIgnored private var yaw = Spring(value: HeadTurn.opening.yaw)
    @ObservationIgnored private var pitch = Spring(value: HeadTurn.opening.pitch)
    @ObservationIgnored private var target: HeadTurn?
    @ObservationIgnored private var lit: [String: Float] = [:]
    @ObservationIgnored private var dragged: CGSize?  // how far the drag has gone, while it goes

    init(set: ElectrodeSet) {
        self.set = set
    }

    var camera: HeadCamera { HeadCamera(size: size) }

    func choose(_ name: String?) {
        chosen = name
        shown = nil
        guard let name, let electrode = set.named(name), let point = HeadSpace.point(of: electrode) else {
            target = nil
            return
        }
        let facing = HeadTurn.facing(simd_normalize(point)).nearest(to: turn)
        if reduceMotion {
            settle(at: facing)
            shown = name
        } else {
            target = facing
        }
    }

    func hover(at location: CGPoint?) {
        let found = location.flatMap { camera.electrode(at: $0, in: set, turned: turn) }
        if found != hovered { hovered = found }
    }

    /// The drag has gone ``translation`` so far: the head turns by what it went since.
    func drag(to translation: CGSize) {
        let since = CGSize(
            width: translation.width - (dragged?.width ?? 0), height: translation.height - (dragged?.height ?? 0))
        dragged = translation
        target = nil
        let turned = HeadTurn(
            yaw: turn.yaw + Float(since.width) * Metrics.Head.turnPerPoint,
            pitch: min(
                max(turn.pitch + Float(since.height) * Metrics.Head.turnPerPoint, HeadTurn.pitches.lowerBound),
                HeadTurn.pitches.upperBound))
        settle(at: turned)
    }

    /// The pointer went up: a click (it hardly moved) chooses what it is on, or nothing.
    func release(at location: CGPoint, after translation: CGSize) {
        dragged = nil
        guard hypot(translation.width, translation.height) <= Metrics.Head.click else { return }
        let found = camera.electrode(at: location, in: set, turned: turn)
        if found != chosen { choose(found) }
    }

    /// Where the dot of the electrode whose text is shown lies in the view, while it faces the
    /// viewer.
    var shownPoint: CGPoint? { shown.flatMap(point(of:)) }

    /// Where the dot of an electrode lies in the view, while it faces the viewer.
    func point(of name: String) -> CGPoint? {
        guard let electrode = set.named(name), let point = HeadSpace.point(of: electrode),
            let normal = HeadSpace.normal(of: electrode), camera.faces(point, normal: normal, turned: turn)
        else { return nil }
        return camera.screen(point, turned: turn)
    }

    /// A frame of ``seconds``: the head turns on, the dots ease, and the scene is drawn so.
    func step(by seconds: Float, in scene: HeadScene) {
        if let target {
            yaw.step(toward: target.yaw, by: seconds)
            pitch.step(toward: target.pitch, by: seconds)
            turn = HeadTurn(yaw: yaw.value, pitch: pitch.value)
            let away = max(abs(yaw.value - target.yaw), abs(pitch.value - target.pitch))
            if shown != chosen, away < Metrics.Head.near { shown = chosen }
            if away < Metrics.Head.settled, abs(yaw.speed) + abs(pitch.speed) < Metrics.Head.settled {
                settle(at: target)
            }
        }
        scene.turn(turn)
        scene.show(chosen: chosen)
        let ease = reduceMotion ? 1 : 1 - exp(-seconds / Metrics.Head.fade)
        for electrode in set.placed {
            let name = electrode.name
            let full = chosen == nil || name == chosen || name == hovered
            let now = lit[name] ?? 1
            let next = now + ((full ? 1 : Float(Palette.Head.dimmed)) - now) * ease
            if next != now || lit[name] == nil {
                lit[name] = next
                scene.light(name, next)
            }
        }
    }

    private func settle(at turned: HeadTurn) {
        target = nil
        yaw = Spring(value: turned.yaw)
        pitch = Spring(value: turned.pitch)
        if turned != turn { turn = turned }
    }
}
