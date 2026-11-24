// The lanes of the timeline, drawn by one canvas for the span of time in view only: a signal or
// a sound as its envelope (earlier layers faded behind the last, as the raw signal behind the
// filtered one; how far a sound was steered faded behind it), values as a line with the area
// between it and their reference tinted by the side it lies on, under a strip of the spans of
// time they are judged against, tinted and named, and a hairline under every row.
import SwiftUI

/// The lanes of the whole session, as wide as it is at the zoom in view.
struct Lanes: View, Equatable {
    let rows: [TimelineRow]
    let pyramids: TrackPyramids
    let start: Double
    let duration: Double
    let pointsPerSecond: Double

    nonisolated static func == (one: Lanes, other: Lanes) -> Bool {
        one.pyramids === other.pyramids && one.start == other.start && one.duration == other.duration
            && one.pointsPerSecond == other.pointsPerSecond && one.rows == other.rows
    }

    var body: some View {
        let width = duration * pointsPerSecond
        let tiles = max(1, Int((width / Metrics.tile).rounded(.up)))
        let tileSeconds = Metrics.tile / pointsPerSecond
        LazyHStack(spacing: 0) {
            ForEach(0..<tiles, id: \.self) { index in
                LaneTile(
                    rows: rows, pyramids: pyramids, start: start + Double(index) * tileSeconds,
                    pointsPerSecond: pointsPerSecond
                )
                .equatable()
                .frame(width: min(Metrics.tile, width - CGFloat(index) * Metrics.tile))
            }
        }
    }
}

/// A span of every lane, from ``start``, drawn once for the zoom.
struct LaneTile: View, Equatable {
    let rows: [TimelineRow]
    let pyramids: TrackPyramids
    let start: Double
    let pointsPerSecond: Double

    nonisolated static func == (one: LaneTile, other: LaneTile) -> Bool {
        one.pyramids === other.pyramids && one.start == other.start && one.pointsPerSecond == other.pointsPerSecond
            && one.rows == other.rows
    }

    var body: some View {
        let scale = TimeScale(leading: start, pointsPerSecond: pointsPerSecond)
        let pyramids = pyramids
        let rows = rows
        Canvas(rendersAsynchronously: true) { context, size in
            let drawing = LaneDrawing(
                scale: scale, from: scale.time(at: -1), to: scale.time(at: size.width + 1), pyramids: pyramids)
            var top: CGFloat = 0
            for row in rows {
                let lane = CGRect(x: 0, y: top, width: size.width, height: row.height)
                drawing.draw(row, in: lane, context: &context)
                context.fill(
                    Path(CGRect(x: 0, y: lane.maxY - Metrics.hairline, width: size.width, height: Metrics.hairline)),
                    with: .style(.separator))
                top += row.height
            }
        }
    }
}

/// One drawing of the lanes: what it draws from, read once.
struct LaneDrawing {
    let scale: TimeScale
    let from: Double
    let to: Double
    let pyramids: TrackPyramids

    func draw(_ row: TimelineRow, in lane: CGRect, context: inout GraphicsContext) {
        switch row.kind {
        case .group:
            context.fill(Path(lane), with: .style(.quinary))
        case .track(.signal(let signal), let colour):
            let layers = pyramids[signal.id] ?? []
            for (index, layer) in layers.enumerated() {
                let opacity = index + 1 < layers.count ? Palette.earlierLayer : 1
                context.fill(
                    envelope(layer, reach: Float(signal.scale), in: lane), with: .color(colour.color.opacity(opacity)))
            }
        case .track(.audio(let audio), let colour):
            context.fill(
                level(Self.steps(of: audio.level, from: from, to: to), in: lane),
                with: .color(colour.color.opacity(Palette.level)))
            if let sound = pyramids[audio.id]?.first {
                context.fill(envelope(sound, reach: 1, in: lane), with: .color(colour.color))
            }
        case .track(.curve(let curve), let colour):
            drawCurve(curve, colour: colour.color, in: lane, context: &context)
        }
    }

    private func drawCurve(_ curve: CurveTrack, colour: Color, in lane: CGRect, context: inout GraphicsContext) {
        var plot = lane
        if !curve.bands.isEmpty {
            let strip = CGRect(x: lane.minX, y: lane.minY, width: lane.width, height: Metrics.strip)
            drawStrip(curve.bands, in: strip, context: &context)
            plot = CGRect(
                x: lane.minX, y: lane.minY + Metrics.Row.strip, width: lane.width,
                height: lane.height - Metrics.Row.strip)
        }
        let steps = Self.steps(of: curve.steps, from: from, to: to)
        for step in steps {
            if let tint = step.colour {
                context.fill(
                    Path(span(step.startS, step.endS, in: plot)), with: .color(tint.color.opacity(Palette.tint)))
            }
        }
        let parts = parts(of: steps, stepped: curve.stepped, in: plot)
        if let reference = curve.reference {
            let y = plot.maxY - CGFloat(reference) * plot.height
            let area = Self.area(of: parts, to: y)
            for (tint, side) in [
                (curve.above, CGRect(x: plot.minX, y: plot.minY, width: plot.width, height: y - plot.minY)),
                (curve.below, CGRect(x: plot.minX, y: y, width: plot.width, height: plot.maxY - y)),
            ] {
                guard let tint else { continue }
                var clipped = context
                clipped.clip(to: Path(side))
                clipped.fill(area, with: .color(tint.color.opacity(Palette.area)))
            }
            var rule = Path()
            rule.move(to: CGPoint(x: 0, y: y))
            rule.addLine(to: CGPoint(x: plot.maxX, y: y))
            context.stroke(rule, with: .style(.secondary), style: Palette.reference)
        }
        var line = Path()
        for part in parts {
            line.addLines(part)
        }
        context.stroke(line, with: .color(colour), lineWidth: Metrics.curveLine)
    }

    /// Spans of time along the top of a lane, tinted, each named at its start where the name
    /// fits in the whole span: drawn by every tile the span crosses, each showing its part, as a
    /// name may cross a tile's edge.
    private func drawStrip(_ bands: [Band], in strip: CGRect, context: inout GraphicsContext) {
        for band in bands where band.endS >= from && band.startS <= to {
            let x = scale.x(of: band.startS)
            let rect = CGRect(x: x, y: strip.minY, width: scale.x(of: band.endS) - x, height: strip.height)
            context.fill(Path(rect), with: .color((band.colour?.color ?? .clear).opacity(Palette.strip)))
            if let title = band.title {
                let text = context.resolve(Text(title).font(.caption).foregroundStyle(.primary))
                if text.measure(in: rect.size).width + 2 * Metrics.bandTitleInset <= rect.width {
                    context.draw(
                        text, at: CGPoint(x: rect.minX + Metrics.bandTitleInset, y: rect.midY), anchor: .leading)
                }
            }
        }
    }

    /// The line through the values of ``steps``, in parts: a step without a value breaks it.
    /// Through the middle of each step, or held from its start to its end (``stepped``).
    func parts(of steps: ArraySlice<CurveStep>, stepped: Bool, in plot: CGRect) -> [[CGPoint]] {
        var parts: [[CGPoint]] = []
        var drawing = false
        for step in steps {
            guard let value = step.value else {
                drawing = false
                continue
            }
            let y = plot.maxY - CGFloat(value) * plot.height
            let start = scale.x(of: step.startS)
            let end = scale.x(of: step.endS)
            let points =
                stepped ? [CGPoint(x: start, y: y), CGPoint(x: end, y: y)] : [CGPoint(x: (start + end) / 2, y: y)]
            if drawing { parts[parts.count - 1] += points } else { parts.append(points) }
            drawing = true
        }
        return parts
    }

    /// The area between each part of a line and the height ``y`` across.
    nonisolated static func area(of parts: [[CGPoint]], to y: CGFloat) -> Path {
        var area = Path()
        for part in parts {
            guard let first = part.first, let last = part.last else { continue }
            area.move(to: CGPoint(x: first.x, y: y))
            for point in part { area.addLine(to: point) }
            area.addLine(to: CGPoint(x: last.x, y: y))
            area.closeSubpath()
        }
        return area
    }

    /// The steps a drawing from ``from`` to ``to`` needs: those in it, and one beyond on each
    /// side, as a line between the middles of two steps crosses the edge between them.
    nonisolated static func steps(of steps: [CurveStep], from: Double, to: Double) -> ArraySlice<CurveStep> {
        let first = steps.partitioningIndex { $0.endS >= from }
        let last = steps.partitioningIndex { $0.startS > to }
        return steps[max(first - 1, 0)..<min(last + 1, steps.endIndex)]
    }

    /// Each step as high as its value, from the bottom of the lane, held until the next.
    func level(_ steps: ArraySlice<CurveStep>, in lane: CGRect) -> Path {
        var area = Path()
        for step in steps {
            guard let value = step.value else { continue }
            let rect = span(step.startS, step.endS, in: lane)
            let height = CGFloat(value) * rect.height
            area.addRect(CGRect(x: rect.minX, y: rect.maxY - height, width: rect.width, height: height))
        }
        return area
    }

    /// The area from the lowest to the highest value of each point of the screen in view: one
    /// shape, its upper edge drawn forward and its lower edge back, broken where the signal is.
    private func envelope(_ pyramid: Pyramid, reach: Float, in lane: CGRect) -> Path {
        let width = Metrics.column / scale.pointsPerSecond  // a column, in seconds
        let columns = pyramid.columns(from: from, to: to, width: width)
        func y(_ value: Float) -> CGFloat { lane.midY - CGFloat(min(max(value / reach, -1), 1)) * lane.height / 2 }
        var path = Path()
        var part = columns.startIndex
        while part < columns.endIndex {
            var end = part + 1  // a part ends where a column is missing: a gap
            while end < columns.endIndex && columns[end].time - columns[end - 1].time < Metrics.gap * width { end += 1 }
            let xs = columns[part..<end].map { scale.x(of: $0.time) }
            path.move(to: CGPoint(x: xs[0], y: y(columns[part].high)))
            for (offset, column) in columns[part..<end].enumerated() {
                path.addLine(to: CGPoint(x: xs[offset], y: y(column.high)))
                path.addLine(to: CGPoint(x: xs[offset] + Metrics.column, y: y(column.high)))
            }
            for (offset, column) in columns[part..<end].enumerated().reversed() {
                let low = max(y(column.low), y(column.high) + Metrics.column)  // a flat signal shows too
                path.addLine(to: CGPoint(x: xs[offset] + Metrics.column, y: low))
                path.addLine(to: CGPoint(x: xs[offset], y: low))
            }
            path.closeSubpath()
            part = end
        }
        return path
    }

    private func span(_ start: Double, _ end: Double, in lane: CGRect) -> CGRect {
        let x = scale.x(of: start)
        return CGRect(x: x, y: lane.minY, width: scale.x(of: end) - x, height: lane.height - Metrics.hairline)
    }
}

extension RandomAccessCollection {
    /// The index of the first element for which ``belongs`` holds, in a collection where it
    /// holds for every element after the first that it holds for.
    func partitioningIndex(where belongs: (Element) -> Bool) -> Index {
        var low = startIndex
        var count = self.count
        while count > 0 {
            let half = count / 2
            let middle = index(low, offsetBy: half)
            if belongs(self[middle]) {
                count = half
            } else {
                low = index(after: middle)
                count -= half + 1
            }
        }
        return low
    }
}
