// The head and the electrodes on it, as RealityKit draws them: the skin in one matte colour
// (HeadSurface.metal), a solid dot of each electrode's colour laid on it (Skin), a ring around
// the one chosen; a still camera before them. The head turns about its middle; nothing else moves.
import Metal
import RealityKit
import SwiftUI

@MainActor
final class HeadScene {
    /// What turns: the skin and the dots on it.
    let root = Entity()
    let camera = PerspectiveCamera()
    private let skin: ModelEntity
    private var dots: [String: Entity] = [:]
    private var rings: [String: Entity] = [:]

    init(set: ElectrodeSet, dotRadius: Float) throws {
        let surface = Skin(of: set.head)
        skin = ModelEntity(
            mesh: try Self.mesh(
                named: "head", points: surface.points, triangles: surface.triangles, normals: surface.normals),
            materials: [])
        root.addChild(skin)
        let ring = dotRadius * Metrics.Head.ringSize
        for electrode in set.placed {
            guard let point = HeadSpace.point(of: electrode), let normal = HeadSpace.normal(of: electrode) else {
                continue
            }
            var material = UnlitMaterial(color: NSColor(set.colours[electrode.name] ?? .gray))
            material.faceCulling = .none  // laid on the skin, either side of it may face the viewer
            let laid = { (inner: Float, outer: Float) in
                let patch = surface.patch(at: point, normal: normal, from: inner, to: outer)
                return ModelEntity(
                    mesh: try Self.mesh(named: electrode.name, points: patch.points, triangles: patch.triangles),
                    materials: [material])
            }
            dots[electrode.name] = try laid(0, dotRadius)
            rings[electrode.name] = try laid(ring * (1 - Metrics.Head.ringWidth), ring)
        }
        for entity in Array(dots.values) + Array(rings.values) { root.addChild(entity) }
        camera.camera.fieldOfViewInDegrees = HeadCamera.fieldOfView
        camera.camera.fieldOfViewOrientation = .vertical
        root.position = SIMD3(0, 0, -HeadCamera.distance)  // before the camera, which looks along -z from the origin
        show(chosen: nil)
    }

    func turn(_ turn: HeadTurn) { root.orientation = turn.rotation }

    /// How lit a dot is: 1 in full, less while another is chosen.
    func light(_ name: String, _ opacity: Float) {
        dots[name]?.components.set(OpacityComponent(opacity: opacity))
    }

    /// The ring around the chosen electrode, and none around the rest.
    func show(chosen: String?) {
        for (name, ring) in rings { ring.isEnabled = name == chosen }
    }

    /// The skin in the colours of the appearance.
    func paint(for scheme: ColorScheme) {
        skin.model?.materials = [Self.skinMaterial(for: scheme)]
    }

    // MARK: -

    static func mesh(
        named name: String, points: [SIMD3<Float>], triangles: [UInt32], normals: [SIMD3<Float>]? = nil
    ) throws -> MeshResource {
        var descriptor = MeshDescriptor(name: name)
        descriptor.positions = MeshBuffers.Positions(points)
        descriptor.normals = normals.map { MeshBuffers.Normals($0) }
        descriptor.primitives = .triangles(triangles)
        return try MeshResource.generate(from: [descriptor])
    }

    /// The matte skin of HeadSurface.metal; plain unlit grey where the shader cannot be had.
    static func skinMaterial(for scheme: ColorScheme) -> any RealityKit.Material {
        let skin = scheme == .dark ? Palette.Head.darkSkin : Palette.Head.lightSkin
        guard let library = MTLCreateSystemDefaultDevice()?.makeDefaultLibrary(),
            var material = try? CustomMaterial(
                surfaceShader: CustomMaterial.SurfaceShader(named: "headSurface", in: library), lightingModel: .unlit)
        else { return UnlitMaterial(color: skin) }
        material.baseColor = .init(tint: skin)
        material.custom.value = SIMD4(scheme == .dark ? Palette.Head.darkRim : Palette.Head.lightRim, 0, 0, 0)
        material.blending = .transparent(opacity: .init(floatLiteral: 1))  // the neck fades out
        material.faceCulling = .back
        return material
    }
}
