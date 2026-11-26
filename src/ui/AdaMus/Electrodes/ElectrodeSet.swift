// The electrodes of a session on the head, as the program describes them: where each lies, in
// the space the head is drawn in, and the colour it is drawn in. The head turns about its middle
// before a still camera: how far it is turned (yaw and pitch), where a point of it then lies on
// the screen, and whether it faces the viewer.
import SwiftUI
import simd

struct ElectrodeSet: Sendable {
    let head: HeadModel
    let electrodes: [Electrode]
    /// The colour of every electrode on the head, by its name.
    let colours: [String: Color]

    init(head: HeadModel, electrodes: [Electrode]) {
        self.head = head
        self.electrodes = electrodes
        colours = ElectrodeColours.of(electrodes)
    }

    /// The electrodes that lie on the head.
    var placed: [Electrode] { electrodes.filter { $0.position != nil } }

    /// Any electrode the session recorded, on the head or not.
    func named(_ name: String) -> Electrode? { electrodes.first { $0.name == name } }
}

/// The space the head is drawn in: metres, y up, the face toward the viewer at +z, the middle of
/// the head at the origin. The program's millimetres have x to the right ear, y to the nose, z up.
enum HeadSpace {
    static let metresPerMillimetre: Float = 0.001
    /// The middle of the head, chin to crown, in the program's millimetres: it turns about here.
    static let middle = SIMD3<Float>(0, -12, -22)

    /// A point of the program's, in the space the head is drawn in.
    static func point(_ mm: SIMD3<Float>) -> SIMD3<Float> { direction(mm - middle) * metresPerMillimetre }
    /// A direction of the program's (x to the right ear, y to the nose, z up), turned into it.
    static func direction(_ v: SIMD3<Float>) -> SIMD3<Float> { SIMD3(-v.x, v.z, v.y) }

    static func point(of electrode: Electrode) -> SIMD3<Float>? {
        electrode.position.map { point(SIMD3($0.map(Float.init))) }
    }

    static func normal(of electrode: Electrode) -> SIMD3<Float>? {
        electrode.normal.map { simd_normalize(direction(SIMD3($0.map(Float.init)))) }
    }
}

/// How far the head is turned: about the vertical (yaw), then toward the viewer (pitch: the
/// crown comes forward, as seen from above).
struct HeadTurn: Equatable, Sendable {
    var yaw: Float
    var pitch: Float

    /// The face, a little from the left and from above.
    static let opening = HeadTurn(yaw: Metrics.Head.openingYaw, pitch: Metrics.Head.openingPitch)
    static let pitches = Metrics.Head.lowestPitch...Metrics.Head.highestPitch

    var rotation: simd_quatf {
        simd_quatf(angle: pitch, axis: SIMD3(1, 0, 0)) * simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
    }

    /// Turned so that a point in the direction ``outward`` faces the viewer, a little below the
    /// middle of the view, as far as the head turns up and down.
    static func facing(_ outward: SIMD3<Float>) -> HeadTurn {
        let yaw = atan2(-outward.x, outward.z)
        let elevation = atan2(outward.y, (outward.x * outward.x + outward.z * outward.z).squareRoot())
        let pitch = min(max(elevation + Metrics.Head.belowMiddle, pitches.lowerBound), pitches.upperBound)
        return HeadTurn(yaw: yaw, pitch: pitch)
    }

    /// The same turn with its yaw the nearest to ``other``'s, a whole turn more or less: the head
    /// turns the short way round.
    func nearest(to other: HeadTurn) -> HeadTurn {
        let turns = ((other.yaw - yaw) / (2 * .pi)).rounded()
        return HeadTurn(yaw: yaw + turns * 2 * .pi, pitch: pitch)
    }
}

/// The still camera the head turns before: where a point of the turned head lies in a view of
/// ``size``, and whether its skin faces the camera.
struct HeadCamera: Sendable {
    static let distance = Metrics.Head.cameraDistance
    static let fieldOfView = Metrics.Head.fieldOfView  // vertical, in degrees
    let size: CGSize

    func screen(_ point: SIMD3<Float>, turned turn: HeadTurn) -> CGPoint? {
        let seen = turn.rotation.act(point)
        let depth = Self.distance - seen.z
        guard depth > 0, size.width > 0, size.height > 0 else { return nil }
        let scale = 1 / tan(Self.fieldOfView * .pi / 360) / depth
        let aspect = Float(size.width / size.height)
        let x = seen.x * scale / aspect
        let y = seen.y * scale
        return CGPoint(x: CGFloat(x + 1) / 2 * size.width, y: CGFloat(1 - y) / 2 * size.height)
    }

    /// Whether the skin at ``point``, facing ``normal``, faces the camera enough to be seen.
    func faces(_ point: SIMD3<Float>, normal: SIMD3<Float>, turned turn: HeadTurn) -> Bool {
        let seen = turn.rotation.act(point)
        let toCamera = simd_normalize(SIMD3(0, 0, Self.distance) - seen)
        return simd_dot(turn.rotation.act(normal), toCamera) > Metrics.Head.seenFacing
    }

    /// The electrode whose point lies nearest ``location``, within reach of a pointer, among
    /// those that face the camera.
    func electrode(at location: CGPoint, in set: ElectrodeSet, turned turn: HeadTurn) -> String? {
        var found: (name: String, distance: CGFloat)?
        for electrode in set.placed {
            guard let point = HeadSpace.point(of: electrode), let normal = HeadSpace.normal(of: electrode),
                faces(point, normal: normal, turned: turn), let at = screen(point, turned: turn)
            else { continue }
            let distance = hypot(at.x - location.x, at.y - location.y)
            if distance <= Metrics.Head.reach, distance < (found?.distance ?? .infinity) {
                found = (electrode.name, distance)
            }
        }
        return found?.name
    }
}

/// A value that follows its target on a critically damped spring: it starts from where it is and
/// how fast it moves, so a new target takes over at any moment, and it never overshoots.
struct Spring: Equatable, Sendable {
    var value: Float
    var speed: Float = 0

    mutating func step(toward target: Float, by seconds: Float) {
        let omega = 2 * Float.pi / Metrics.Head.response
        speed += (-omega * omega * (value - target) - 2 * omega * speed) * seconds
        value += speed * seconds
    }
}

/// The colour of each electrode on the head: the system colours that mean nothing else in the
/// app (HIG "Color": orange and indigo are focus and rest, purple and teal the focus and the EEG,
/// pink the music, red and green a result) while there are no more of them than colours; more
/// take hues spread over the whole wheel, so that the next one along always looks different. The
/// reference is grey.
enum ElectrodeColours {
    static let system: [Color] = [.blue, .mint, .yellow, .brown, .cyan]

    static func of(_ electrodes: [Electrode]) -> [String: Color] {
        let recorded = electrodes.filter { $0.kind == .recorded && $0.position != nil }
        var colours = [String: Color]()
        if recorded.count <= system.count {
            for (electrode, colour) in zip(recorded, system) { colours[electrode.name] = colour }
        } else {
            for (index, electrode) in inRings(recorded).enumerated() {
                colours[electrode.name] = Color(
                    hue: (Double(index) * Metrics.Head.goldenTurn).truncatingRemainder(dividingBy: 1),
                    saturation: Metrics.Head.saturation,
                    brightness: index.isMultiple(of: 2) ? Metrics.Head.brighter : Metrics.Head.darker)
            }
        }
        for electrode in electrodes where electrode.kind == .reference { colours[electrode.name] = .gray }
        return colours
    }

    /// In rings from the crown down, each ring around the head in turn: neighbours follow one
    /// another.
    static func inRings(_ electrodes: [Electrode]) -> [Electrode] {
        func ring(_ e: Electrode) -> Int { Int(((e.position?[2] ?? 0) / Metrics.Head.ringHeight).rounded()) }
        func around(_ e: Electrode) -> Double { atan2(e.position?[0] ?? 0, e.position?[1] ?? 0) }
        return electrodes.sorted { ring($0) != ring($1) ? ring($0) > ring($1) : around($0) < around($1) }
    }
}
