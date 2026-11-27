// What the program says through its API: the models of src/adamus/server/api_models.py, decoded
// with their snake_case keys turned into camelCase.
import Foundation

// MARK: - Sessions

struct SessionSummary: Decodable, Identifiable, Hashable {
    let id: String
    let name: String
}

/// What a session is to be called from now on.
struct SessionRename: Encodable {
    let name: String
}

struct SessionPage: Decodable {
    let total: Int
    let sessions: [SessionSummary]
}

// MARK: - Values

/// A colour of the palette, by the name of its hue; what it stands for, the program says.
enum Colour: String, Decodable, Hashable, Sendable {
    case orange, indigo, gray, teal, pink, purple
}

enum Format: String, Decodable, Sendable {
    case text, count, decimal, percent, duration, datetime, frequency
}

struct Value: Decodable, Hashable, Sendable {
    enum Content: Hashable, Sendable {
        case text(String)
        case number(Double)
        case unknown
    }

    let content: Content
    let format: Format

    init(_ content: Content, format: Format) {
        self.content = content
        self.format = format
    }

    private enum Key: String, CodingKey { case value, format }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Key.self)
        format = try container.decode(Format.self, forKey: .format)
        if try container.decodeNil(forKey: .value) {
            content = .unknown
        } else if let number = try? container.decode(Double.self, forKey: .value) {
            content = .number(number)
        } else {
            content = .text(try container.decode(String.self, forKey: .value))
        }
    }
}

struct LegendEntry: Decodable, Hashable, Sendable {
    let colour: Colour
    let title: String
}

// MARK: - Timeline

struct Band: Decodable, Hashable, Sendable {
    let startS: Double
    let endS: Double
    let title: String?
    let colour: Colour?
}

struct CurveStep: Decodable, Hashable, Sendable {
    let startS: Double
    let endS: Double
    let value: Double?
    let colour: Colour?
}

struct SignalTrack: Decodable, Hashable, Sendable {
    let id: String
    let title: String
    let caption: String?
    let scale: Double
    let rateHz: Double
    let samples: Int
    let layers: Int
}

struct AudioTrack: Decodable, Hashable, Sendable {
    let id: String
    let title: String
    let caption: String?
    let startS: Double
    let durationS: Double
    let rateHz: Double
    let buckets: Int
    let level: [CurveStep]  // drawn faded behind the sound, each held until the next
}

struct CurveTrack: Decodable, Hashable, Sendable {
    let id: String
    let title: String
    let caption: String?
    let steps: [CurveStep]
    let reference: Double?
    let above: Colour?  // of the area between the line and the reference, where it is over it
    let below: Colour?  // and where it is under it
    let bands: [Band]  // in a strip along the top
    let stepped: Bool
}

enum Track: Decodable, Identifiable, Hashable, Sendable {
    case signal(SignalTrack)
    case audio(AudioTrack)
    case curve(CurveTrack)

    private enum Key: String, CodingKey { case kind }
    private enum Kind: String, Decodable { case signal, audio, curve }

    init(from decoder: Decoder) throws {
        switch try decoder.container(keyedBy: Key.self).decode(Kind.self, forKey: .kind) {
        case .signal: self = .signal(try SignalTrack(from: decoder))
        case .audio: self = .audio(try AudioTrack(from: decoder))
        case .curve: self = .curve(try CurveTrack(from: decoder))
        }
    }

    var id: String {
        switch self {
        case .signal(let track): track.id
        case .audio(let track): track.id
        case .curve(let track): track.id
        }
    }

    var title: String {
        switch self {
        case .signal(let track): track.title
        case .audio(let track): track.title
        case .curve(let track): track.title
        }
    }

    var caption: String? {
        switch self {
        case .signal(let track): track.caption
        case .audio(let track): track.caption
        case .curve(let track): track.caption
        }
    }
}

struct TrackGroup: Decodable, Hashable, Sendable {
    let title: String
    let colour: Colour
    let tracks: [Track]
}

enum Severity: String, Decodable, Sendable {
    case debug, info, warning, error
}

struct ConsoleLine: Decodable, Hashable, Sendable {
    let timeS: Double
    let source: String?
    let severity: Severity
    let text: String
}

struct Console: Decodable, Hashable, Sendable {
    let title: String
    let lines: [ConsoleLine]
}

struct Timeline: Decodable, Sendable {
    let startS: Double
    let endS: Double
    let groups: [TrackGroup]
    let legend: [LegendEntry]
    let consoles: [Console]
}

/// An electrode of a session: where it lies on the skin of the head and which way the skin faces
/// there (millimetres, in the frame of HeadModel), or nowhere, and what it records.
struct Electrode: Decodable, Identifiable, Hashable, Sendable {
    enum Kind: String, Decodable, Sendable {
        case recorded
        case reference  // what the others are measured against
    }

    let name: String
    let kind: Kind
    let position: [Double]?
    let normal: [Double]?
    let about: String

    var id: String { name }
}

/// The head the electrodes are drawn on, in millimetres: x to the right ear, y to the nose, z up.
struct HeadModel: Decodable, Sendable {
    let vertices: [Float]  // x, y and z of every point
    let triangles: [UInt32]  // three points each, the skin facing outward
}

/// A part of the timeline, cut out of the data of the session (JSON).
struct Excerpt: Decodable, Sendable {
    let text: String
}

// MARK: - Dashboard

struct Fact: Decodable, Hashable, Sendable {
    let label: String
    let value: Value
    let status: Status?  // how the session went, where the fact tells it
}

enum Status: String, Decodable, Sendable {
    case good, bad
}

struct FactTable: Decodable, Hashable, Sendable {
    let label: String
    let columns: [String]
    let rows: [[Value]]
}

enum SectionItem: Decodable, Hashable, Sendable {
    case fact(Fact)
    case table(FactTable)

    private enum Key: String, CodingKey { case kind }
    private enum Kind: String, Decodable { case fact, table }

    init(from decoder: Decoder) throws {
        switch try decoder.container(keyedBy: Key.self).decode(Kind.self, forKey: .kind) {
        case .fact: self = .fact(try Fact(from: decoder))
        case .table: self = .table(try FactTable(from: decoder))
        }
    }
}

struct FactSection: Decodable, Hashable, Sendable {
    let title: String
    let items: [SectionItem]
}

struct Highlight: Decodable, Hashable, Sendable {
    let title: String
    let value: Value
    let caption: String
    let colour: Colour
}

struct Dashboard: Decodable, Sendable {
    let name: String
    let problem: String?
    let pending: String?
    let summary: [Fact]  // what the session is and how it ended
    let failures: [String]  // what went wrong, each on its own
    let caution: String?  // what bears on how its results are read
    let highlights: [Highlight]
    let song: Song?  // the source of its music, as it kept it; none if its music never began
    let sections: [FactSection]
}

/// A stretch of a song, or of the music of a session, that belongs to one state.
struct SongPart: Decodable, Hashable, Sendable {
    let startS: Double
    let endS: Double
    let mark: String
    let title: String
    let colour: Colour
}

// MARK: - A new session

struct SourceChoice: Decodable, Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let live: Bool
    let recordings: [String]
}

struct NewSessionForm: Decodable, Sendable {
    let sources: [SourceChoice]
    let sourcesHint: String
    let namePattern: String
    let nameHint: String
}

struct SourceCheck: Decodable, Equatable, Sendable {
    let warning: String?
    let problem: String?
}

struct AudioOutputChoice: Decodable, Identifiable, Hashable, Sendable {
    let id: String
    let title: String
}

struct AudioOutputs: Decodable, Sendable {
    let outputs: [AudioOutputChoice]
    /// The one the system plays on, chosen at first; none when it plays on none.
    let system: String?
    let hint: String
}

struct SessionRequest: Encodable, Equatable, Sendable {
    let sourceId: String
    let recording: String?
    /// The song the music of the session is made of.
    let songId: String
    let name: String
    /// The audio output the music is to play on.
    let output: String
}

enum Phase: String, Decodable, Sendable {
    case idle, starting, running, finishing
}

struct RunState: Decodable, Equatable, Sendable {
    let run: Int
    let phase: Phase
    let sessionId: String?
    /// Why the session could not start, or ended in an error.
    let problem: String?
}

// MARK: - The library of songs

/// What a part of a song can be marked.
struct MarkChoice: Decodable, Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let colour: Colour
}

/// A song of the library, or a draft of one, where it is marked.
struct Song: Decodable, Identifiable, Hashable, Sendable, SoundShape {
    let id: String
    let title: String
    let artist: String?
    let lengthS: Double
    let loudness: [Double]  // of equal slices of the song, the loudest 1
    let parts: [SongPart]  // what is marked, in the order of the song
    let cover: String?  // the version of its picture, which changes with it; none without one
}

struct SongPage: Decodable {
    let total: Int
    let songs: [Song]  // by title
}

struct SongMarks: Decodable, Sendable {
    let marks: [MarkChoice]
    let unmarked: String  // what a part no mark covers is called
}

/// What a draft that goes into the library is called, and by whom (nil, left out, when it is not
/// known), and the path of the picture chosen for it (nil keeps its own).
struct DraftKept: Encodable, Sendable {
    let title: String
    let artist: String?
    let picture: String?
}

struct DraftOfFile: Encodable, Sendable {
    let path: String
}

/// Mark the stretch from ``startS`` to ``endS``; nil, left out, unmarks it.
struct SongMark: Encodable, Equatable, Sendable {
    let startS: Double
    let endS: Double
    let mark: String?
}
