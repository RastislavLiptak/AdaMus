// A program of the tests in place of the real one: it answers each call of the app with what a
// test gave it for that call, and keeps every call, so that a test can see what the app asked.
// The models of the window are tested with it in the cases the real program cannot be brought
// into at will: a refusal, an answer still being worked on, a list that changed meanwhile.
import Foundation
import Synchronization
import Testing

final class StubProgram: Sendable {
    /// A call of the app, as the program received it.
    struct Call: Equatable, Sendable {
        let route: String  // its method and path, as "GET api/sessions"
        let query: [String: String]
        let token: String?
        let body: Data?
    }

    private struct Answer: Sendable {
        let status: Int
        let body: Data
    }

    private struct Calls: Sendable {
        var answers: [String: [Answer]] = [:]
        var made: [Call] = []
    }

    private static let registered: Bool = {
        URLProtocol.registerClass(StubProtocol.self)
        return true
    }()
    /// The stand-in programs by the host of their address: each test has its own.
    private static let programs = Mutex<[String: StubProgram]>([:])

    let api: APIClient
    private let calls = Mutex(Calls())

    init() {
        precondition(Self.registered)
        let host = "\(UUID().uuidString.lowercased()).stub"
        api = APIClient(address: URL(string: "http://\(host)")!, token: "stub-token")
        Self.programs.withLock { $0[host] = self }
    }

    /// Answer the next call of ``route`` with ``json``; after the last answer given, every call
    /// gets that one again.
    func answer(_ route: String, status: Int = 200, _ json: String) {
        calls.withLock { $0.answers[route, default: []].append(Answer(status: status, body: Data(json.utf8))) }
    }

    var made: [Call] { calls.withLock { $0.made } }

    func made(_ route: String) -> [Call] { made.filter { $0.route == route } }

    fileprivate static func program(for request: URLRequest) -> StubProgram? {
        guard let host = request.url?.host() else { return nil }
        return programs.withLock { $0[host] }
    }

    fileprivate func respond(to request: URLRequest) -> (status: Int, body: Data) {
        let url = request.url!
        let route = "\(request.httpMethod ?? "GET") \(url.path().dropFirst())"
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        let call = Call(
            route: route, query: Dictionary(uniqueKeysWithValues: items.map { ($0.name, $0.value ?? "") }),
            token: request.value(forHTTPHeaderField: APIClient.tokenHeader), body: Self.body(of: request))
        let answer = calls.withLock { calls -> Answer? in
            calls.made.append(call)
            guard let answers = calls.answers[route], let first = answers.first else { return nil }
            if answers.count > 1 { calls.answers[route]?.removeFirst() }
            return first
        }
        guard let answer else { return (404, Data(#"{"detail": "No answer for \#(route)."}"#.utf8)) }
        return (answer.status, answer.body)
    }

    /// A body sent through a session comes as a stream.
    private static func body(of request: URLRequest) -> Data? {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let read = stream.read(&buffer, maxLength: buffer.count)
            guard read > 0 else { break }
            data.append(buffer, count: read)
        }
        return data
    }

    /// Wait until ``holds``, at most a few seconds.
    @MainActor
    static func until(_ what: String, _ holds: () -> Bool) async throws {
        let end = ContinuousClock.now + .seconds(5)
        while !holds() {
            guard ContinuousClock.now < end else {
                Issue.record("waited in vain until \(what)")
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}

/// What the system asks to load the calls of the app to a stand-in program.
private final class StubProtocol: URLProtocol {
    override class func canInit(with request: URLRequest) -> Bool { StubProgram.program(for: request) != nil }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let program = StubProgram.program(for: request), let url = request.url else { return }
        let (status, body) = program.respond(to: request)
        let response = HTTPURLResponse(
            url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
