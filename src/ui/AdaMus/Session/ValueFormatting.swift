// How a value is written: with the formatters of Foundation, in the person's locale; durations
// in the unit that suits their size; an unknown value as a dash.
import Foundation

enum ValueFormatting {
    static let unknown = "—"

    static func text(_ value: Value, locale: Locale = .current, timeZone: TimeZone = .current) -> String {
        switch value.content {
        case .unknown: return unknown
        case .text(let text): return text
        case .number(let number): return text(number, as: value.format, locale: locale, timeZone: timeZone)
        }
    }

    static func text(
        _ number: Double, as format: Format, locale: Locale = .current, timeZone: TimeZone = .current
    ) -> String {
        switch format {
        case .text: return number.formatted(.number.locale(locale))
        case .count: return number.formatted(.number.precision(.fractionLength(0)).locale(locale))
        case .decimal: return number.formatted(.number.precision(.fractionLength(2)).locale(locale))
        case .percent: return number.formatted(.percent.precision(.fractionLength(0)).locale(locale))
        case .duration: return duration(number, locale: locale)
        case .datetime:
            return Date(timeIntervalSince1970: number).formatted(
                Date.FormatStyle(locale: locale, timeZone: timeZone).year().month().day().hour().minute())
        case .frequency:
            return Measurement(value: number, unit: UnitFrequency.hertz)
                .formatted(
                    .measurement(width: .abbreviated, numberFormatStyle: .number.precision(.fractionLength(1))).locale(
                        locale))
        }
    }

    /// Below a second in milliseconds, below a minute in seconds, as the locale writes the unit;
    /// then as a clock: 1:05, 1:02:05. The unit is chosen by the value as written, so that
    /// 59.96 s is 1:00, not 60.0 s.
    static func duration(_ seconds: Double, locale: Locale = .current) -> String {
        guard seconds.isFinite else { return unknown }
        let milliseconds = (seconds * 1000).rounded()
        if abs(milliseconds) < 1000 {
            return Duration.milliseconds(milliseconds).formatted(
                .units(allowed: [.milliseconds], width: .abbreviated).locale(locale))
        }
        if abs((seconds * 10).rounded()) < 600 {  // in tenths of a second
            return Duration.milliseconds(milliseconds).formatted(
                .units(allowed: [.seconds], width: .abbreviated, fractionalPart: .show(length: 1)).locale(locale))
        }
        return clock(seconds)
    }

    /// A moment of a session as a clock: 1:05, 1:02:05; with ``tenths``, 1:05.3, the tenths after
    /// the decimal separator of the locale. Written directly, as the ruler writes many of them
    /// while it scrolls.
    static func clock(_ seconds: Double, tenths: Bool = false, locale: Locale = .current) -> String {
        guard seconds.isFinite else { return unknown }
        let unit = tenths ? 10.0 : 1.0
        let units = Int((abs(seconds) * unit).rounded())
        let whole = tenths ? units / 10 : units
        let sign = seconds < 0 && units > 0 ? "−" : ""
        let hours = whole / 3600
        let minutes = (whole % 3600) / 60
        let rest = whole % 60
        var text = hours > 0 ? "\(hours):\(twoDigits(minutes)):\(twoDigits(rest))" : "\(minutes):\(twoDigits(rest))"
        if tenths { text += (locale.decimalSeparator ?? ".") + "\(units % 10)" }
        return sign + text
    }

    private static func twoDigits(_ number: Int) -> String {
        number < 10 ? "0\(number)" : "\(number)"
    }
}
