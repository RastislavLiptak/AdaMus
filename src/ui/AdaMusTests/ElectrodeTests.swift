// The electrodes on the head: where a point of the program lies in the space the head is drawn
// in, how the head turns to face an electrode, where a dot then lies on the screen and whether
// it faces the viewer, the colours, the spring, and what the head of the overview does.
import Foundation
import SwiftUI
import Testing
import simd

/// A head of a few points, and electrodes on it, as the program would send them (millimetres:
/// x to the right ear, y to the nose, z up).
enum SmallHead {
    static let head = HeadModel(
        vertices: [0, 100, -22, 80, -12, -22, -80, -12, -22, 0, -12, 90], triangles: [0, 1, 3, 0, 3, 2, 1, 2, 3])

    static func electrode(_ name: String, _ at: [Double], _ normal: [Double], kind: Electrode.Kind = .recorded)
        -> Electrode
    {
        Electrode(name: name, kind: kind, position: at, normal: normal, about: "\(name) records something.")
    }

    /// AF7 on the left of the forehead, AF8 on the right, TP9 behind the left ear, Fpz the
    /// reference in the middle of the forehead, AUX nowhere on the head.
    static let set = ElectrodeSet(
        head: head,
        electrodes: [
            electrode("TP9", [-80, -40, -40], [-0.9, -0.4, -0.1]),
            electrode("AF7", [-50, 70, 0], [-0.6, 0.8, 0]),
            electrode("AF8", [50, 70, 0], [0.6, 0.8, 0]),
            Electrode(name: "AUX", kind: .recorded, position: nil, normal: nil, about: "AUX is not on the head."),
            electrode("Fpz", [0, 90, 0], [0, 1, 0], kind: .reference),
        ])
}

struct HeadSpaceTests {
    @Test func theNoseLooksAtTheViewerTheRightEarToTheLeftTheCrownUpTheMiddleAtTheOrigin() {
        #expect(HeadSpace.point(HeadSpace.middle) == .zero)
        let nose = HeadSpace.point(HeadSpace.middle + SIMD3(0, 100, 0))
        let rightEar = HeadSpace.point(HeadSpace.middle + SIMD3(100, 0, 0))
        let crown = HeadSpace.point(HeadSpace.middle + SIMD3(0, 0, 100))
        #expect(nose == SIMD3(0, 0, 0.1) && rightEar == SIMD3(-0.1, 0, 0) && crown == SIMD3(0, 0.1, 0))
        #expect(HeadSpace.normal(of: SmallHead.set.named("Fpz")!) == SIMD3(0, 0, 1))
    }

    @Test func onlyTheElectrodesOnTheHeadArePlaced() {
        #expect(SmallHead.set.placed.map(\.name) == ["TP9", "AF7", "AF8", "Fpz"])
        #expect(SmallHead.set.named("AUX") == nil && SmallHead.set.named("AF7")?.name == "AF7")
    }
}

struct HeadTurnTests {
    @Test(arguments: ["TP9", "AF7", "AF8", "Fpz"])
    func aTurnFacingAnElectrodeBringsItToTheViewerALittleBelowTheMiddle(name: String) throws {
        let electrode = try #require(SmallHead.set.named(name))
        let point = try #require(HeadSpace.point(of: electrode))
        let seen = HeadTurn.facing(simd_normalize(point)).rotation.act(simd_normalize(point))
        #expect(abs(seen.x) < 1e-5, "in the middle across")
        #expect(abs(seen.y + sin(Metrics.Head.belowMiddle)) < 1e-5 && seen.z > 0, "toward the viewer, a little low")
    }

    @Test func theHeadTurnsUpAndDownOnlySoFar() {
        #expect(HeadTurn.facing(SIMD3(0, 1, 0)).pitch == Metrics.Head.highestPitch)  // the crown
        #expect(HeadTurn.facing(SIMD3(0, -1, 0.01)).pitch == Metrics.Head.lowestPitch)  // under the chin
    }

    @Test func theHeadTurnsTheShortWayRound() {
        let from = HeadTurn(yaw: 3, pitch: 0)
        let to = HeadTurn(yaw: -3, pitch: 0).nearest(to: from)
        #expect(abs(to.yaw - (2 * .pi - 3)) < 1e-5 && abs(to.yaw - from.yaw) < .pi)
    }

    @Test func aSpringReachesItsTargetWithoutOvershooting() {
        var spring = Spring(value: 0)
        var highest: Float = 0
        for _ in 0..<240 {
            spring.step(toward: 1, by: 1 / 120)
            highest = max(highest, spring.value)
        }
        #expect(abs(spring.value - 1) < 1e-3 && highest <= 1)
    }
}

struct HeadCameraTests {
    private let camera = HeadCamera(size: CGSize(width: 400, height: 300))

    @Test func theMiddleOfTheHeadIsTheMiddleOfTheViewAndRightIsRightUpIsUp() throws {
        let still = HeadTurn(yaw: 0, pitch: 0)
        #expect(camera.screen(.zero, turned: still) == CGPoint(x: 200, y: 150))
        let right = try #require(camera.screen(SIMD3(0.05, 0, 0), turned: still))
        let up = try #require(camera.screen(SIMD3(0, 0.05, 0), turned: still))
        #expect(right.x > 200 && right.y == 150 && up.y < 150 && up.x == 200)
        // the vertical field of view: a point at its edge, at the distance of the middle, lies on the edge
        let edge = HeadCamera.distance * tan(HeadCamera.fieldOfView * .pi / 360)
        #expect(abs((camera.screen(SIMD3(0, edge, 0), turned: still)?.y ?? 1) - 0) < 1e-3)
    }

    @Test func aPointBehindTheCameraIsNowhereOnTheScreen() {
        #expect(camera.screen(SIMD3(0, 0, HeadCamera.distance + 0.1), turned: HeadTurn(yaw: 0, pitch: 0)) == nil)
        #expect(HeadCamera(size: .zero).screen(.zero, turned: .opening) == nil)
    }

    @Test func theSkinFacingTheViewerIsSeenTheBackOfTheHeadIsNot() {
        let still = HeadTurn(yaw: 0, pitch: 0)
        #expect(camera.faces(SIMD3(0, 0, 0.09), normal: SIMD3(0, 0, 1), turned: still))
        #expect(!camera.faces(SIMD3(0, 0, -0.09), normal: SIMD3(0, 0, -1), turned: still))
        #expect(camera.faces(SIMD3(0, 0, -0.09), normal: SIMD3(0, 0, -1), turned: HeadTurn(yaw: .pi, pitch: 0)))
        // skin that turns away sideways, as at the edge of the head, is not seen enough to be chosen
        #expect(!camera.faces(SIMD3(0.09, 0, 0), normal: SIMD3(1, 0, 0), turned: still))
    }

    @Test func thePointerChoosesTheDotItIsOnOrNearOnlyAmongThoseSeen() throws {
        let set = SmallHead.set
        let turn = HeadTurn.facing(simd_normalize(HeadSpace.point(of: set.named("Fpz")!)!))
        let fpz = try #require(camera.screen(HeadSpace.point(of: set.named("Fpz")!)!, turned: turn))
        #expect(camera.electrode(at: fpz, in: set, turned: turn) == "Fpz")
        let near = CGPoint(x: fpz.x + Metrics.Head.reach - 1, y: fpz.y)
        let far = CGPoint(x: fpz.x + Metrics.Head.reach + 1, y: fpz.y)
        #expect(camera.electrode(at: near, in: set, turned: turn) == "Fpz")
        #expect(camera.electrode(at: far, in: set, turned: turn) == nil)
        let behind = HeadTurn(yaw: .pi, pitch: 0)  // the face away: its dots are not chosen
        let whereFpzWas = try #require(camera.screen(HeadSpace.point(of: set.named("Fpz")!)!, turned: behind))
        #expect(camera.electrode(at: whereFpzWas, in: set, turned: behind) == nil)
    }
}

struct SkinTests {
    private let skin = Skin(of: SmallHead.head)
    /// The corners of the small head where it is drawn: the tip of the nose, the right and the
    /// left side, the crown.
    private let corners = stride(from: 0, to: SmallHead.head.vertices.count, by: 3).map {
        HeadSpace.point(SIMD3(SmallHead.head.vertices[$0...$0 + 2]))
    }

    /// Whether ``point`` lies on the skin of ``head``: on the plane of one of its triangles,
    /// within it.
    private func liesOnTheSkin(_ point: SIMD3<Float>, of head: HeadModel) -> Bool {
        stride(from: 0, to: head.triangles.count, by: 3).contains { corner in
            let (a, b, c) = (
                corners[Int(head.triangles[corner])], corners[Int(head.triangles[corner + 1])],
                corners[Int(head.triangles[corner + 2])]
            )
            let across = simd_normalize(simd_cross(b - a, c - a))
            guard abs(simd_dot(point - a, across)) < 1e-6 else { return false }
            return [(a, b), (b, c), (c, a)].allSatisfy {
                simd_dot(simd_cross($0.1 - $0.0, point - $0.0), across) > -1e-7
            }
        }
    }

    @Test func aRingIsItsCirclesFromTheInnerToTheOuterEdgeFilledBetween() {
        let ring = Skin.ring(from: 0.001, to: 0.003)
        let count = Metrics.Head.ringSegments
        #expect(ring.points.count == (Metrics.Head.patchBands + 1) * count)
        #expect(ring.points.allSatisfy { $0.y == 0 })
        #expect(ring.points.prefix(count).allSatisfy { abs(simd_length($0) - 0.001) < 1e-7 })
        #expect(ring.points.suffix(count).allSatisfy { abs(simd_length($0) - 0.003) < 1e-7 })
        let area = stride(from: 0, to: ring.triangles.count, by: 3).map {
            let (a, b, c) = (
                ring.points[Int(ring.triangles[$0])], ring.points[Int(ring.triangles[$0 + 1])],
                ring.points[Int(ring.triangles[$0 + 2])]
            )
            return simd_length(simd_cross(b - a, c - a)) / 2
        }
        .reduce(0, +)
        #expect(abs(area - .pi * (0.003 * 0.003 - 0.001 * 0.001)) < 0.01 * area, "covered once, no gap")
    }

    /// On a ridge of the small head, across it: a flat dot would hang over both sides of it.
    /// The small head is wound inward, the head of the program outward: either way.
    @Test(arguments: [[0, 3], [1, 3]], [false, true])
    func aDotLaidWhereTheSkinBendsLiesJustOffItOnBothSides(ridge: [Int], woundOutward: Bool) {
        let triangles = stride(from: 0, to: SmallHead.head.triangles.count, by: 3).flatMap {
            let (a, b, c) = (
                SmallHead.head.triangles[$0], SmallHead.head.triangles[$0 + 1], SmallHead.head.triangles[$0 + 2]
            )
            return woundOutward ? [a, c, b] : [a, b, c]
        }
        let head = HeadModel(vertices: SmallHead.head.vertices, triangles: triangles)
        let place = (corners[ridge[0]] + corners[ridge[1]]) / 2
        let normal = simd_normalize(place - corners.reduce(.zero, +) / Float(corners.count))  // out from the middle
        let dot = Skin(of: head).patch(at: place, normal: normal, from: 0, to: 0.02)
        let flat = Skin.ring(from: 0, to: 0.02)
        let lying = simd_quatf(from: SIMD3(0, 1, 0), to: normal)
        for (laid, point) in zip(dot.points, flat.points) {
            let moved = laid - place - lying.act(point)
            #expect(simd_length(simd_cross(moved, normal)) < 1e-6, "moved only along the normal")
            // the skin under it is at most 35 mm off the flat dot, across the head 78 mm and more
            #expect(simd_length(moved) < 0.05, "onto the skin under it, not that across the head")
            #expect(liesOnTheSkin(laid - normal * Metrics.Head.offSkin, of: head), "just off the skin, \(laid)")
        }
        #expect(dot.triangles == flat.triangles)
    }

    @Test func whereNoSkinLiesUnderItADotStaysFlatJustOffWhereItWouldBe() {
        let place = SIMD3<Float>(1, 0, 0)
        let normal = SIMD3<Float>(1, 0, 0)
        let lying = simd_quatf(from: SIMD3(0, 1, 0), to: normal)
        let dot = skin.patch(at: place, normal: normal, from: 0.001, to: 0.002)
        let flat = Skin.ring(from: 0.001, to: 0.002).points.map {
            place + lying.act($0) + normal * Metrics.Head.offSkin
        }
        #expect(zip(dot.points, flat).allSatisfy { simd_distance($0, $1) < 1e-7 })
    }

    @Test func theNormalOfTheSkinAtEachPointIsWholeAndLeansAsTheTrianglesAroundIt() {
        let normals = skin.normals
        #expect(normals.count == corners.count && normals.allSatisfy { abs(simd_length($0) - 1) < 1e-5 })
        for corner in stride(from: 0, to: skin.triangles.count, by: 3) {
            let (a, b, c) = (
                Int(skin.triangles[corner]), Int(skin.triangles[corner + 1]), Int(skin.triangles[corner + 2])
            )
            let face = simd_normalize(simd_cross(corners[b] - corners[a], corners[c] - corners[a]))
            #expect([a, b, c].allSatisfy { simd_dot(normals[$0], face) > -1e-6 }, "never against a triangle of its")
        }
        // at the crown the triangles around it, the larger weighing more, add up to straight down
        #expect(simd_distance(normals[3], SIMD3(0, -1, 0)) < 1e-5)
    }
}

struct ElectrodeColourTests {
    @Test func upToFiveElectrodesTakeTheSystemColoursInTheirOrderTheReferenceGrey() {
        let colours = SmallHead.set.colours
        #expect(colours["TP9"] == .blue && colours["AF7"] == .mint && colours["AF8"] == .yellow)
        #expect(colours["Fpz"] == .gray && colours["AUX"] == nil)
    }

    @Test func moreElectrodesEachTakeAColourTheNextOneAlongADifferentOne() {
        let names = (0..<64).map { "E\($0)" }
        let electrodes = names.enumerated().map { index, name in
            let angle = Double(index % 8) / 8 * 2 * .pi
            return SmallHead.electrode(name, [80 * cos(angle), 80 * sin(angle), Double(index / 8) * 15], [0, 0, 1])
        }
        let colours = ElectrodeColours.of(electrodes)
        #expect(colours.count == 64 && Set(colours.values).count == 64)
        let ordered = ElectrodeColours.inRings(electrodes).map { colours[$0.name]! }
        #expect(zip(ordered, ordered.dropFirst()).allSatisfy { $0 != $1 })
        #expect(ElectrodeColours.inRings(electrodes).first?.position?[2] == 105)  // the crown first
    }
}

@MainActor
struct HeadTurnerTests {
    private func turner() throws -> (HeadTurner, HeadScene) {
        let turner = HeadTurner(set: SmallHead.set)
        turner.size = CGSize(width: 400, height: 300)
        return (turner, try HeadScene(set: SmallHead.set, dotRadius: Metrics.Head.dot))
    }

    private func settle(_ turner: HeadTurner, in scene: HeadScene) {
        for _ in 0..<240 { turner.step(by: 1 / 120, in: scene) }
    }

    @Test func aChosenElectrodeIsSaidOnceTheHeadNearlyFacesIt() throws {
        let (turner, scene) = try turner()
        turner.choose("AF8")
        #expect(turner.chosen == "AF8" && turner.shown == nil, "not yet: the head still turns")
        turner.step(by: 1 / 120, in: scene)
        #expect(turner.shown == nil)
        settle(turner, in: scene)
        #expect(turner.shown == "AF8")
        let facing = HeadTurn.facing(simd_normalize(HeadSpace.point(of: SmallHead.set.named("AF8")!)!))
        #expect(abs(turner.turn.yaw - facing.yaw) < 0.01 && abs(turner.turn.pitch - facing.pitch) < 0.01)
        #expect(turner.shownPoint != nil, "its dot, where the line goes")
        turner.choose(nil)
        #expect(turner.chosen == nil && turner.shown == nil && turner.shownPoint == nil)
    }

    @Test func withReduceMotionTheHeadTurnsAtOnce() throws {
        let (turner, _) = try turner()
        turner.reduceMotion = true
        turner.choose("TP9")
        #expect(turner.shown == "TP9")
        #expect(turner.turn == HeadTurn.facing(simd_normalize(HeadSpace.point(of: SmallHead.set.named("TP9")!)!)))
    }

    @Test func aDragTurnsTheHeadByWhatItWentAndStopsATurnUnderWay() throws {
        let (turner, scene) = try turner()
        turner.choose("AF7")
        turner.step(by: 1 / 120, in: scene)
        let before = turner.turn
        turner.drag(to: CGSize(width: 10, height: 0))
        turner.drag(to: CGSize(width: 30, height: 5000))  // far down: the pitch stops at its highest
        #expect(abs(turner.turn.yaw - (before.yaw + 30 * Metrics.Head.turnPerPoint)) < 1e-4)
        #expect(turner.turn.pitch == Metrics.Head.highestPitch)
        turner.drag(to: CGSize(width: 30, height: -5000))  // far up: at its lowest
        #expect(turner.turn.pitch == Metrics.Head.lowestPitch)
        let dragged = turner.turn
        settle(turner, in: scene)
        #expect(turner.turn == dragged, "the turn toward AF7 was let go")
    }

    @Test func aClickChoosesTheDotUnderItAndAClickOnNoDotLetsGo() throws {
        let (turner, scene) = try turner()
        turner.choose("Fpz")
        settle(turner, in: scene)
        let af8 = try #require(turner.point(of: "AF8"))
        turner.release(at: af8, after: .zero)
        #expect(turner.chosen == "AF8")
        turner.release(at: CGPoint(x: 1, y: 1), after: CGSize(width: Metrics.Head.click + 1, height: 0))
        #expect(turner.chosen == "AF8", "a drag is no click")
        turner.release(at: CGPoint(x: 1, y: 1), after: .zero)
        #expect(turner.chosen == nil)
    }

    @Test func theDotUnderThePointerIsHovered() throws {
        let (turner, scene) = try turner()
        turner.choose("Fpz")
        settle(turner, in: scene)
        let fpz = try #require(turner.point(of: "Fpz"))
        turner.hover(at: fpz)
        #expect(turner.hovered == "Fpz")
        turner.hover(at: nil)
        #expect(turner.hovered == nil)
    }
}
