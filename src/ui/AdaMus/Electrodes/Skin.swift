// The skin of the head in the space it is drawn in, and what lies on it: a dot or a ring laid
// along the shape of the skin, not flat across it. Flat, its edge would sink under the skin
// wherever the head curves away from it or its faces meet at an angle.
import simd

struct Skin: Sendable {
    let points: [SIMD3<Float>]
    let triangles: [UInt32]
    private let faces: [Face]
    /// No point of a triangle lies farther than this from its corners.
    private let longestEdge: Float
    /// A line this nearly along a triangle crosses it nowhere that counts.
    private static let parallel: Float = 1e-12

    private struct Face: Sendable {
        let a, b, c: SIMD3<Float>
    }

    /// A flat shape about to be laid: its points and the triangles between them.
    struct Patch: Sendable {
        let points: [SIMD3<Float>]
        let triangles: [UInt32]
    }

    init(of head: HeadModel) {
        let points = stride(from: 0, to: head.vertices.count, by: 3).map {
            HeadSpace.point(SIMD3(head.vertices[$0], head.vertices[$0 + 1], head.vertices[$0 + 2]))
        }
        let faces = stride(from: 0, to: head.triangles.count, by: 3).map {
            Face(
                a: points[Int(head.triangles[$0])], b: points[Int(head.triangles[$0 + 1])],
                c: points[Int(head.triangles[$0 + 2])])
        }
        self.points = points
        self.faces = faces
        triangles = head.triangles
        let edges = faces.map { max(simd_distance($0.a, $0.b), simd_distance($0.b, $0.c), simd_distance($0.c, $0.a)) }
        longestEdge = edges.max() ?? 0
    }

    /// The normal of the skin at each of its points: of the triangles around it, the larger
    /// weighing more.
    var normals: [SIMD3<Float>] {
        var normals = [SIMD3<Float>](repeating: .zero, count: points.count)
        for (corner, face) in zip(stride(from: 0, to: triangles.count, by: 3), faces) {
            let across = simd_cross(face.b - face.a, face.c - face.a)
            for index in triangles[corner..<corner + 3] { normals[Int(index)] += across }
        }
        return normals.map { simd_length($0) > 0 ? simd_normalize($0) : $0 }
    }

    /// A ring from ``inner`` to ``outer`` (a disc from 0) about ``place``, across ``normal``: each
    /// of its points moved along the normal onto the skin, then ``Metrics.Head.offSkin`` off it.
    /// Where no skin lies under a point, it stays flat.
    func patch(at place: SIMD3<Float>, normal: SIMD3<Float>, from inner: Float, to outer: Float) -> Patch {
        let flat = Self.ring(from: inner, to: outer)
        let lying = simd_quatf(from: SIMD3(0, 1, 0), to: normal)
        // The skin a point of it meets is no farther from the place than across the ring, and
        // a triangle of that skin has a corner no farther than its longest edge from there.
        let near = faces.filter { simd_distance($0.a, place) <= 2 * outer + longestEdge }
        let points = flat.points.map { point in
            let laid = place + lying.act(point)
            let along = near.compactMap { Self.crossing(from: laid, along: normal, $0) }.min { abs($0) < abs($1) }
            return laid + normal * ((along ?? 0) + Metrics.Head.offSkin)
        }
        return Patch(points: points, triangles: flat.triangles)
    }

    /// A flat ring across y: its circles from ``inner`` to ``outer``, each of
    /// ``Metrics.Head.ringSegments`` points, joined circle to circle.
    static func ring(from inner: Float, to outer: Float) -> Patch {
        let count = Metrics.Head.ringSegments
        let circles = Metrics.Head.patchBands + 1
        var points = [SIMD3<Float>]()
        for circle in 0..<circles {
            let radius = inner + (outer - inner) * Float(circle) / Float(circles - 1)
            for step in 0..<count {
                let angle = Float(step) / Float(count) * 2 * .pi
                points.append(SIMD3(cos(angle), 0, sin(angle)) * radius)
            }
        }
        var triangles = [UInt32]()
        for circle in 0..<circles - 1 {
            for step in 0..<count {
                let (a, c) = (circle * count + step, circle * count + (step + 1) % count)
                let (b, d) = (a + count, c + count)
                triangles += [a, b, d, a, d, c].map(UInt32.init)
            }
        }
        return Patch(points: points, triangles: triangles)
    }

    /// How far along ``direction`` from ``origin`` the line through them crosses ``face``, either
    /// way (Möller and Trumbore).
    private static func crossing(from origin: SIMD3<Float>, along direction: SIMD3<Float>, _ face: Face) -> Float? {
        let (first, second) = (face.b - face.a, face.c - face.a)
        let across = simd_cross(direction, second)
        let determinant = simd_dot(first, across)
        guard abs(determinant) > parallel else { return nil }
        let from = origin - face.a
        let u = simd_dot(from, across) / determinant
        let turned = simd_cross(from, first)
        let v = simd_dot(direction, turned) / determinant
        guard u >= 0, v >= 0, u + v <= 1 else { return nil }
        return simd_dot(second, turned) / determinant
    }
}
