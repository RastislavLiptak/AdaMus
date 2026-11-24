// The colours the program names, as the system colours of the same hue, which adapt to light
// and dark and to Increase Contrast by themselves (HIG "Color"), and how the timeline draws
// with them; the symbol and colour of how a session went.
import SwiftUI

enum Palette {
    static let tint = 0.3  // of a span of time behind what is drawn over it
    static let earlierLayer = 0.3  // of a layer of a signal behind the last one
    static let level = 0.3  // of how far a sound was steered, behind the sound
    static let area = 0.45  // between a line and its reference
    static let strip = 0.6  // of the spans of time along the top of a lane
    static let reference = StrokeStyle(lineWidth: Metrics.referenceLine, dash: Metrics.referenceDash)

    /// The skin of the head the electrodes are drawn on: a light grey on light, a dark one on
    /// dark, its outline darker on light and lighter on dark (as a share of the skin).
    enum Head {
        static let lightSkin = NSColor(srgbRed: 0.902, green: 0.902, blue: 0.918, alpha: 1)
        static let darkSkin = NSColor(srgbRed: 0.282, green: 0.282, blue: 0.306, alpha: 1)
        static let lightRim: Float = 0.72
        static let darkRim: Float = 1.9
        static let dimmed = 0.35  // the dots not chosen, while one is
    }
}

extension Colour {
    var color: Color {
        switch self {
        case .orange: .orange
        case .indigo: .indigo
        case .gray: .gray
        case .teal: .teal
        case .pink: .pink
        case .purple: .purple
        }
    }
}

extension Status {
    var systemName: String {
        switch self {
        case .good: "checkmark.circle.fill"
        case .bad: "xmark.octagon.fill"
        }
    }

    var color: Color {
        switch self {
        case .good: .green
        case .bad: .red
        }
    }
}
