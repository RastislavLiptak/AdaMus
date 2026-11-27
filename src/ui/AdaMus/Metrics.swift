// The sizes and distances of the window, in points, in one place: what the system does not give
// itself (its controls, its forms and its sheets set their own).
import CoreGraphics

enum Metrics {
    /// Between elements, from the tightest (a line of text and its caption) to the widest (the
    /// parts of a page).
    enum Spacing {
        static let line: CGFloat = 2
        static let tight: CGFloat = 4
        static let symbol: CGFloat = 6  // a symbol and its text
        static let small: CGFloat = 8
        static let regular: CGFloat = 12
        static let large: CGFloat = 16
        static let page: CGFloat = 20  // the parts of a page, and its margin
        static let sheet: CGFloat = 32  // around what a sheet shows alone
    }

    static let window = CGSize(width: 1280, height: 800)  // when it first opens
    static let sidebar = (min: CGFloat(180), ideal: CGFloat(240), max: CGFloat(360))
    static let inspector = (min: CGFloat(300), ideal: CGFloat(440), max: CGFloat(640))
    static let sheetWidth: CGFloat = 480
    static let songInForm: CGFloat = 220  // the widest the song of a new session goes, beside its title
    static let library = CGSize(width: 960, height: 600)  // the window, when it first opens
    static let songEditor: CGFloat = 560
    static let markedPartSheet = CGSize(width: 640, height: 560)
    static let checklistWidth: CGFloat = 560
    static let lastWords = CGSize(width: 640, height: 240)  // what a program that ended printed
    static let cornerRadius: CGFloat = 8  // of a box of text

    /// A song or the music of a session: its cover and its sound track.
    enum Sound {
        static let cover: CGFloat = 140
        static let coverShadow: CGFloat = 6
        static let playButton: CGFloat = 56  // round, on the middle of the cover
        static let track: CGFloat = 44
        static let bar: CGFloat = 3  // the width of a bar of the track, whatever its width
        static let barGap: CGFloat = 2
        static let grab: CGFloat = 12  // beside the track, where a drag along it may begin
        static let hitRegion: CGFloat = 28  // the least height of a button of the menu over a stretch
        static let leastBar: CGFloat = 0.08  // a silent slice still shows, this high of the track
        static let unplayed = 0.38  // how strongly what is still to play shows
        static let playhead: CGFloat = 2.5
        static let step = 1.0  // s, the track moved by VoiceOver
        static let coverRounding: CGFloat = 0.06  // of the side of a cover, its corners
        static let noteShare: CGFloat = 0.4  // of the side of a cover without a picture, its note
        static let thumbnail: CGFloat = 40  // a cover in a list
        static let magnet: CGFloat = 8  // a drag's end this near the edge of a part clings to it
        static let markingTrack: CGFloat = 96  // the track where the parts are marked
        static let zoomStep = 2.0
        static let mostZoom = 64.0
        static let selected = 0.35  // how strongly a stretch selected shows over the track
        static let leastDrag: CGFloat = 3  // a press moved less selects nothing: a click
    }

    static let summaryColumn: CGFloat = 180  // the narrowest a fact of the overview goes
    static let resultBox: CGFloat = 170  // the narrowest a main result goes

    static let logTime: CGFloat = 54
    static let logSource: CGFloat = 64

    static let tick: CGFloat = 6  // of the ruler
    static let tickLabel = CGVector(dx: 3, dy: -2)  // from the tick, and from the middle
    static let playhead: CGFloat = 2
    static let leastMarked: CGFloat = 4  // the narrowest part a drag with Shift marks
    static let hairline: CGFloat = 1

    /// The rows of the timeline, each as high whatever else is shown or folded.
    enum Row {
        static let group: CGFloat = 24
        static let curve: CGFloat = 56
        static let strip = Metrics.strip + Metrics.Spacing.line  // what a strip adds to a curve
        static let signal: CGFloat = 80  // a signal or the sound
    }

    static let trackNames: CGFloat = 160  // the column of the names of the tracks
    static let ruler: CGFloat = 28
    static let tile: CGFloat = 512  // of the lanes, drawn at once
    static let column: CGFloat = 1  // of an envelope: a point of the screen
    static let gap: CGFloat = 1.5  // columns apart that break an envelope
    static let bandTitleInset: CGFloat = 4  // of the title of a span, from its edges
    static let strip: CGFloat = 14  // the spans of time along the top of a curve
    static let curveLine: CGFloat = 1.5
    static let referenceLine: CGFloat = 1  // dashed, across a chart at its reference
    static let referenceDash: [CGFloat] = [3, 3]
    static let tickSpacing: CGFloat = 72  // the least between two labelled times of the ruler

    /// The head the electrodes are drawn on, and how it turns.
    enum Head {
        static let box: CGFloat = 380  // the height of the box of the electrodes on the overview
        static let popover = CGSize(width: 300, height: 180)  // of the head in the popover of a track
        static let popoverText: CGFloat = 150  // the most of its text shown before it scrolls
        static let textGap: CGFloat = 32  // between the head and its text, where the line runs
        static let line: CGFloat = 2  // from the name to the electrode
        static let lineGap: CGFloat = 12  // the line stops this short of the name and of the point
        static let reach: CGFloat = 14  // how near a point the pointer chooses it
        static let click: CGFloat = 4  // the most a click moves; more is a drag
        static let appear = 0.18  // s, the text of the chosen comes in
        static let appearShift: CGFloat = 8  // from this far to the right

        static let cameraDistance: Float = 0.6  // m, from the middle of the head
        static let fieldOfView: Float = 28  // degrees, vertical: the whole head, chin to crown
        static let openingYaw: Float = -0.18  // rad: the face, a little from the left
        static let openingPitch: Float = 0.3  // and from above
        static let lowestPitch: Float = -0.6
        static let highestPitch: Float = 1.0
        static let belowMiddle: Float = 0.1  // rad: a chosen electrode sits this far below the middle
        static let seenFacing: Float = 0.15  // the least cosine of a skin that is seen
        static let turnPerPoint: Float = 0.01  // rad a drag turns the head per point
        static let response: Float = 0.35  // s, of the spring that turns it (HIG motion: brief)
        static let near: Float = 0.12  // rad from facing, where the text of the chosen comes
        static let settled: Float = 0.002  // rad, and as little speed: it has stopped
        static let fade: Float = 0.12  // s, a point lights up under the pointer, and dims again

        static let dot: Float = 0.004  // m, the radius of a point on the skin
        static let popoverDot: Float = 0.0065  // larger, as the head in the popover is smaller
        static let ringSize: Float = 2.4  // times the dot, around the chosen one
        static let ringWidth: Float = 0.25  // of the ring, as a share of its radius
        static let offSkin: Float = 0.0004  // m, a point lies this far off the skin, not in it
        static let ringSegments = 64  // points on each circle of a dot or a ring
        static let patchBands = 3  // circles between its inner and outer edge, laid each on the skin

        static let goldenTurn = 0.381966  // of the colour wheel, from one electrode to the next
        static let saturation = 0.78
        static let brighter = 0.92
        static let darker = 0.72
        static let ringHeight = 15.0  // mm, the height of a ring of electrodes around the head
    }
}
