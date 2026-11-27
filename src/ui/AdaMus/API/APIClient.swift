// The API of the program on this computer. Every call carries the token the program said when
// it was ready, so that no other program can use it.
import Foundation

struct APIError: LocalizedError, Equatable {
    let message: String
    let status: Int?

    var errorDescription: String? { message }
}

struct APIClient: Sendable, Equatable {
    static let tokenHeader = "X-AdaMus-Token"
    static let pageSize = 200
    /// How soon the program is asked again of what it is still doing.
    static let lookAgain = Duration.milliseconds(500)

    let address: URL
    let token: String

    /// How every answer is read; the tests read what the program sends with it too.
    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()

    // MARK: Sessions

    /// Every session, a page at a time.
    func sessions() async throws -> [SessionSummary] {
        try await everyPage { offset in
            let page: SessionPage = try await decoded("api/sessions", query: page(from: offset))
            return (page.sessions, page.total)
        }
    }

    func dashboard(of session: String) async throws -> Dashboard {
        try await decoded("api/sessions/\(session)/dashboard")
    }

    func timeline(of session: String) async throws -> Timeline {
        try await decoded("api/sessions/\(session)/timeline")
    }

    /// The head the electrodes are drawn on.
    func head() async throws -> HeadModel {
        try await decoded("api/head")
    }

    func electrodes(of session: String) async throws -> [Electrode] {
        try await decoded("api/sessions/\(session)/electrodes")
    }

    /// A part of the timeline of a session, cut out of its data.
    func excerpt(of session: String, from start: Double, to end: Double) async throws -> Excerpt {
        try await decoded("api/sessions/\(session)/excerpt", query: ["start_s": "\(start)", "end_s": "\(end)"])
    }

    /// The samples of a track, as the program sends them: 32-bit floats of this computer.
    func samples(of track: String, in session: String) async throws -> [Float] {
        let data = try await body("api/sessions/\(session)/tracks/\(track)")
        let count = data.count / MemoryLayout<Float>.stride
        return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            initialized = data.copyBytes(to: buffer) / MemoryLayout<Float>.stride
        }
    }

    /// Where the sound of an audio track is played from; a player cannot send a header, so the
    /// token goes in the address.
    func soundURL(of track: String, in session: String) -> URL {
        url("api/sessions/\(session)/tracks/\(track)/sound", query: ["token": token])
    }

    /// The session called ``name`` from now on; its id changes with its name.
    func rename(_ session: String, to name: String) async throws -> SessionSummary {
        try await decoded(
            "api/sessions/\(session)", method: "PATCH", sending: try Self.encoder.encode(SessionRename(name: name)))
    }

    func delete(_ session: String) async throws {
        _ = try await body("api/sessions/\(session)", method: "DELETE")
    }

    // MARK: A new session

    func newSessionForm() async throws -> NewSessionForm {
        try await decoded("api/new-session")
    }

    /// The form again, once the headset has been looked for.
    func lookForHeadset() async throws -> NewSessionForm {
        try await decoded("api/new-session/refresh", method: "POST")
    }

    /// Where the song a session kept is played from, and its picture, if it carries one.
    func song(ofSession session: String, _ song: Song) -> (sound: URL, cover: URL?) {
        files(of: "api/sessions/\(session)/song", song)
    }

    func check(source: String, recording: String?) async throws -> SourceCheck {
        try await decoded("api/sources/\(source)/check", query: recording.map { ["recording": $0] } ?? [:])
    }

    /// The audio outputs of the Mac now, and the one the system plays on.
    func audioOutputs() async throws -> AudioOutputs {
        try await decoded("api/audio-outputs")
    }

    func start(_ request: SessionRequest) async throws -> RunState {
        try await decoded("api/run", method: "POST", sending: try Self.encoder.encode(request))
    }

    func runState() async throws -> RunState {
        try await decoded("api/run")
    }

    func stopRun() async throws {
        _ = try await body("api/run/stop", method: "POST")
    }

    // MARK: The library of songs

    /// Every song, a page at a time.
    func songs() async throws -> [Song] {
        try await everyPage { offset in
            let page: SongPage = try await decoded("api/songs", query: page(from: offset))
            return (page.songs, page.total)
        }
    }

    func songMarks() async throws -> SongMarks {
        try await decoded("api/marks")
    }

    func deleteSong(_ song: Song.ID) async throws {
        _ = try await body("api/songs/\(song)", method: "DELETE")
    }

    /// Where a song is played from and its picture, if it carries one.
    func song(_ song: Song) -> (sound: URL, cover: URL?) {
        files(of: "api/songs/\(song.id)", song)
    }

    // MARK: Drafts, where a song is changed

    /// A draft of the song file at ``path``, which the program analyses: seconds.
    func draft(ofFile path: String) async throws -> Song {
        try await decoded("api/drafts", method: "POST", sending: try Self.encoder.encode(DraftOfFile(path: path)))
    }

    func draft(of song: Song.ID) async throws -> Song {
        try await decoded("api/songs/\(song)/draft", method: "POST")
    }

    func mark(_ draft: Song.ID, _ marked: SongMark) async throws -> Song {
        try await decoded("api/drafts/\(draft)/marks", method: "POST", sending: try Self.encoder.encode(marked))
    }

    /// Put the draft into the library: a new song, or in place of the song it is a draft of.
    func keep(_ draft: Song.ID, as kept: DraftKept) async throws -> Song {
        try await decoded("api/drafts/\(draft)/keep", method: "POST", sending: try Self.encoder.encode(kept))
    }

    func discard(_ draft: Song.ID) async throws {
        _ = try await body("api/drafts/\(draft)", method: "DELETE")
    }

    func draftFiles(_ draft: Song) -> (sound: URL, cover: URL?) {
        files(of: "api/drafts/\(draft.id)", draft)
    }

    // MARK: -

    /// With the token in their address, as a player and an image ask without a header; the
    /// picture with its version too, so that one changed is read anew, not kept as it was.
    private func files(of path: String, _ song: Song) -> (sound: URL, cover: URL?) {
        let cover = song.cover.map { url("\(path)/cover", query: ["token": token, "version": $0]) }
        return (url("\(path)/sound", query: ["token": token]), cover)
    }

    private func page(from offset: Int) -> [String: String] {
        ["offset": "\(offset)", "limit": "\(Self.pageSize)"]
    }

    /// Every item of a list read a page at a time: items may come or go meanwhile, and pages
    /// move, so an item read already is not listed again, and a page with nothing new ends it.
    private func everyPage<Item: Identifiable>(
        _ read: (Int) async throws -> (items: [Item], total: Int)
    ) async throws -> [Item] {
        var page = try await read(0)
        var listed = page.items
        while listed.count < page.total {
            page = try await read(listed.count)
            let known = Set(listed.map(\.id))
            let new = page.items.filter { !known.contains($0.id) }
            if new.isEmpty { break }
            listed += new
        }
        return listed
    }

    private func decoded<T: Decodable>(
        _ path: String, query: [String: String] = [:], method: String = "GET", sending: Data? = nil
    ) async throws -> T {
        try Self.decoder.decode(T.self, from: try await body(path, query: query, method: method, sending: sending))
    }

    private func body(
        _ path: String, query: [String: String] = [:], method: String = "GET", sending: Data? = nil
    ) async throws -> Data {
        var request = URLRequest(url: url(path, query: query))
        request.httpMethod = method
        request.setValue(token, forHTTPHeaderField: Self.tokenHeader)
        if let sending {
            request.httpBody = sending
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode
        guard let status, (200..<300).contains(status) else {
            throw APIError(message: Self.detail(of: data) ?? "AdaMus did not answer as expected.", status: status)
        }
        return data
    }

    /// What the program says went wrong: FastAPI puts it in ``detail``, as text, or as a list of
    /// what was wrong with a request, each with its message.
    private static func detail(of data: Data) -> String? {
        struct Said: Decodable { let detail: String }
        struct Wrong: Decodable { let msg: String }
        struct Listed: Decodable { let detail: [Wrong] }
        if let said = try? JSONDecoder().decode(Said.self, from: data) { return said.detail }
        return (try? JSONDecoder().decode(Listed.self, from: data))?.detail.map(\.msg).joined(separator: " ")
    }

    private func url(_ path: String, query: [String: String]) -> URL {
        var url = address.appending(path: path)
        if !query.isEmpty {
            url.append(queryItems: query.sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) })
        }
        return url
    }
}
