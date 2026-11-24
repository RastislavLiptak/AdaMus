// A signal too long to draw sample by sample, drawn as a recording app draws sound: the lowest
// and highest value of each column of the screen. The lowest and highest value of every bucket
// of samples are kept at every size of bucket, each four times the one below, so that drawing
// any span at any zoom reads about as many buckets as it draws columns.

/// The pyramids of the signals and the sound of a timeline, by the track, built once: compared
/// by identity, so that a view drawn from them knows at once whether they are the same.
final class TrackPyramids: Sendable {
    let byTrack: [String: [Pyramid]]

    init(_ byTrack: [String: [Pyramid]] = [:]) {
        self.byTrack = byTrack
    }

    subscript(track: String) -> [Pyramid]? { byTrack[track] }
}

/// The lowest and highest value of a column of a plot, which begins at ``time``.
struct EnvelopePoint: Hashable, Sendable {
    let time: Double
    let low: Float
    let high: Float
}

struct Pyramid: Sendable {
    /// Buckets of ``size`` samples: the lowest and the highest value of each.
    struct Level: Sendable {
        let size: Int
        let low: [Float]
        let high: [Float]
    }

    static let factor = 4
    static let edge = 1e-9  // of a column: what rounding may take off a time at its start

    let rate: Double
    let start: Double
    /// The samples themselves, the finest level; a sound has a lowest and a highest of its own.
    let samples: Level
    let levels: [Level]

    /// A signal: one value at each sample.
    init(samples: [Float], rate: Double, start: Double) {
        self.init(low: samples, high: samples, rate: rate, start: start)
    }

    /// A sound, as its envelope: the lowest and the highest value at each of its buckets.
    init(low: [Float], high: [Float], rate: Double, start: Double) {
        precondition(low.count == high.count && rate > 0)
        self.rate = rate
        self.start = start
        samples = Level(size: 1, low: low, high: high)
        var levels: [Level] = []
        var below = samples
        while below.low.count > Self.factor {  // up to the level of a few buckets
            below = Self.coarser(below)
            levels.append(below)
        }
        self.levels = levels
    }

    /// The lowest and highest value of each column from ``from`` to ``to`` seconds, the columns
    /// ``width`` seconds wide and fixed on the time, so that they do not change as the plot
    /// scrolls. A column of a gap (no number in it) is left out.
    func columns(from: Double, to: Double, width: Double) -> [EnvelopePoint] {
        guard width > 0, to > from else { return [] }
        let level = levels.last { Double($0.size) <= width * rate } ?? samples
        let bucket = Double(level.size) / rate
        let first = max(0, Int(((from - start) / bucket).rounded(.down)))
        let last = min(level.low.count, Int(((to - start) / bucket).rounded(.up)) + 1)
        guard first < last else { return [] }
        var points: [EnvelopePoint] = []
        points.reserveCapacity(Int((to - from) / width) + 2)
        var column = Int.min
        var lowest = Float.infinity
        var highest = -Float.infinity
        level.low.withUnsafeBufferPointer { lows in
            level.high.withUnsafeBufferPointer { highs in
                for index in first..<last {
                    let time = start + Double(index) * bucket
                    // a bucket at the edge of a column belongs to it, also when rounding says
                    // a hair less
                    let at = Int((time / width + Self.edge).rounded(.down))
                    if at != column {
                        if lowest <= highest {
                            points.append(EnvelopePoint(time: Double(column) * width, low: lowest, high: highest))
                        }
                        column = at
                        lowest = .infinity
                        highest = -.infinity
                    }
                    let low = lows[index]
                    let high = highs[index]
                    if low < lowest { lowest = low }  // NaN compares false: a gap is skipped
                    if high > highest { highest = high }
                }
            }
        }
        if lowest <= highest {
            points.append(EnvelopePoint(time: Double(column) * width, low: lowest, high: highest))
        }
        return points
    }

    /// The level above ``level``: each of its buckets the lowest and highest of four.
    private static func coarser(_ level: Level) -> Level {
        let count = (level.low.count + factor - 1) / factor
        var low = [Float](repeating: .infinity, count: count)
        var high = [Float](repeating: -.infinity, count: count)
        level.low.withUnsafeBufferPointer { lows in
            level.high.withUnsafeBufferPointer { highs in
                for index in 0..<lows.count {
                    let bucket = index / factor
                    if lows[index] < low[bucket] { low[bucket] = lows[index] }
                    if highs[index] > high[bucket] { high[bucket] = highs[index] }
                }
            }
        }
        return Level(size: level.size * factor, low: low, high: high)
    }
}
