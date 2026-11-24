// The lines a pipe receives, each as soon as it has come whole. The handler of a file handle is
// called whenever data arrive, so a line printed and flushed is read at once. A line ends with a
// line feed, or with a carriage return, as a tool that shows its progress in place ends it.
import Foundation

enum Lines {
    static func of(_ handle: FileHandle) -> AsyncStream<String> {
        AsyncStream { continuation in
            continuation.onTermination = { _ in handle.readabilityHandler = nil }  // read no more
            let pending = Pending()
            handle.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty {  // the end: the writer has closed the pipe
                    handle.readabilityHandler = nil
                    if let rest = pending.rest() { continuation.yield(rest) }
                    continuation.finish()
                    return
                }
                for line in pending.lines(adding: data) { continuation.yield(line) }
            }
        }
    }

    /// What has come of a line that has not ended yet. The handler is called one call after
    /// another, never at once, so it needs no lock.
    private final class Pending: @unchecked Sendable {
        private var data = Data()

        private static let lineFeed = UInt8(ascii: "\n")
        private static let carriageReturn = UInt8(ascii: "\r")

        func lines(adding more: Data) -> [String] {
            data.append(more)
            var lines: [String] = []
            while let end = data.firstIndex(where: { $0 == Self.lineFeed || $0 == Self.carriageReturn }) {
                var next = data.index(after: end)
                if data[end] == Self.carriageReturn {
                    if next == data.endIndex { break }  // a line feed may still come, of the same end
                    if data[next] == Self.lineFeed { next = data.index(after: next) }
                }
                lines.append(String(decoding: data[data.startIndex..<end], as: UTF8.self))
                data.removeSubrange(data.startIndex..<next)
            }
            return lines
        }

        func rest() -> String? {
            let line = data.last == Self.carriageReturn ? data.dropLast() : data
            return line.isEmpty ? nil : String(decoding: line, as: UTF8.self)
        }
    }
}
