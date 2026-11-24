// The program itself, in Python, which the app starts and talks to (src/adamus/app_link.py).
// It prints a line of JSON for each message: the checklist of its start as it changes, then the
// address and token of its API. Whatever else it prints goes to its error output, of which the
// last lines are kept to say why it ended, if it ends on its own. It ends when its input
// closes, which happens when the app ends in any way.
import Foundation
import Observation

/// A check of the start of the program.
struct Check: Decodable, Hashable, Identifiable, Sendable {
    enum Status: String, Decodable, Sendable {
        case pending, running, passed, failed, skipped
    }

    let name: String
    let status: Status
    let detail: String
    let fix: String?

    var id: String { name }
}

/// A line the program printed to the app.
enum ProgramMessage: Decodable, Equatable {
    case checks([Check])
    case ready(address: URL, token: String)

    private enum Key: String, CodingKey { case checks, ready }
    private struct Ready: Decodable {
        let address: URL
        let token: String
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Key.self)
        if let checks = try container.decodeIfPresent([Check].self, forKey: .checks) {
            self = .checks(checks)
        } else {
            let ready = try container.decode(Ready.self, forKey: .ready)
            self = .ready(address: ready.address, token: ready.token)
        }
    }

    init?(line: String) {
        guard let message = try? JSONDecoder().decode(Self.self, from: Data(line.utf8)) else { return nil }
        self = message
    }
}

@MainActor @Observable
final class Program {
    enum State: Equatable {
        /// The app lies outside a project folder, where there is no program to start.
        case notInProject(URL)
        /// The checks of the start, as far as they have got.
        case starting([Check])
        case ready(APIClient)
        /// A check failed: the checklist says which, and how to fix it.
        case failed([Check])
        /// The program ended on its own, with the last lines it printed.
        case ended(String)
    }

    static let errorLinesKept = 40
    static let lastWords = Duration.seconds(1)  // what is printed after the end of a program

    /// Wait for ``work`` at most ``limit``; whether it was done.
    private static func within(_ limit: Duration, _ work: @escaping @Sendable () async -> Void) async -> Bool {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await work()
                return true
            }
            group.addTask {
                try? await Task.sleep(for: limit)
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
    }

    private(set) var state: State
    /// The last lines the program printed to its error output, read by no view as they come.
    @ObservationIgnored private var errorLines: [String] = []
    /// The executable of the program and the folder it runs in; none for a program the app did
    /// not start, which it only talks to.
    private let command: (executable: URL, folder: URL)?
    private var process: Process?  // held while it runs
    /// The input of the program that runs: held open, as the system closes it when the app ends,
    /// and the program ends with it.
    private var input: Pipe?

    init(executable: URL, folder: URL) {
        command = (executable, folder)
        let found = FileManager.default.isExecutableFile(atPath: executable.path(percentEncoded: false))
        state = found ? .starting([]) : .notInProject(folder)
    }

    /// A program that runs already, which the app neither starts nor ends.
    init(running api: APIClient) {
        command = nil
        state = .ready(api)
    }

    /// The program of the project folder the app lies in.
    static func ofProject(_ folder: URL) -> Program {
        Program(executable: folder.appending(path: ".venv/bin/adamus"), folder: folder)
    }

    /// The program of the app: that of the project folder it lies in. A development build that
    /// the tests of the window start is given instead the program they started, which answers
    /// at `ADAMUS_PROGRAM_ADDRESS` to calls with `ADAMUS_PROGRAM_TOKEN`.
    static func ofApp(in folder: URL) -> Program {
        #if DEBUG
            let given = ProcessInfo.processInfo.environment
            if let address = given["ADAMUS_PROGRAM_ADDRESS"].flatMap(URL.init(string:)),
                let token = given["ADAMUS_PROGRAM_TOKEN"]
            {
                return Program(running: APIClient(address: address, token: token))
            }
        #endif
        return ofProject(folder)
    }

    /// Start the program; again after it ended.
    func start() {
        guard process == nil, let command else { return }
        if case .notInProject = state { return }
        state = .starting([])
        errorLines = []
        let process = Process()
        let input = Pipe()
        let output = Pipe()
        let errors = Pipe()
        process.executableURL = command.executable
        process.currentDirectoryURL = command.folder
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors
        let exit = AsyncStream<(Process.TerminationReason, Int32)> { ended in
            process.terminationHandler = { finished in
                ended.yield((finished.terminationReason, finished.terminationStatus))
                ended.finish()
            }
        }
        do {
            try process.run()
        } catch {
            state = .ended(error.localizedDescription)
            return
        }
        self.process = process
        self.input = input
        let messages = Task { await read(output.fileHandleForReading) }
        let said = Task { await keep(errors.fileHandleForReading) }
        Task {
            // How it ended, once what it printed has been read: its last message may come after
            // its end. Not longer than a moment, as a process it started may hold its output;
            // then they read no more, not to write into the next run.
            for await (reason, status) in exit {
                _ = await Self.within(Self.lastWords) {
                    await messages.value
                    await said.value
                }
                messages.cancel()
                said.cancel()
                ended(reason: reason, status: status)
            }
        }
    }

    private func read(_ output: FileHandle) async {
        for await line in Lines.of(output) {
            switch ProgramMessage(line: line) {
            case .checks(let checks): state = .starting(checks)
            case .ready(let address, let token): state = .ready(APIClient(address: address, token: token))
            case nil: continue  // not a message: nothing else is printed there
            }
        }
    }

    /// How the program ended: with a status of its own, or stopped by the system, which says
    /// nothing else of it (as when macOS stops a program that used what it may not).
    nonisolated static func why(_ reason: Process.TerminationReason, _ status: Int32) -> String {
        switch reason {
        case .uncaughtSignal: "AdaMus was stopped by the system: \(String(cString: strsignal(status)))."
        case .exit: "AdaMus ended with status \(status)."
        @unknown default: "AdaMus ended with status \(status)."
        }
    }

    private func keep(_ errors: FileHandle) async {
        for await line in Lines.of(errors) {
            errorLines.append(line)
            if errorLines.count > Self.errorLinesKept { errorLines.removeFirst() }
        }
    }

    private func ended(reason: Process.TerminationReason, status: Int32) {
        process = nil
        input = nil
        switch state {
        case .starting(let checks) where checks.contains { $0.status == .failed }:
            state = .failed(checks)
        default:
            state = .ended((errorLines + [Self.why(reason, status)]).joined(separator: "\n"))
        }
    }
}
