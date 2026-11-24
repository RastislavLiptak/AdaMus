// AdaMus: one window with the sessions in a sidebar and the one chosen beside it. A single
// window, so the app quits when it closes (HIG "Windows").
import SwiftUI

@main
struct AdaMusApp: App {
    @State private var program = Program.ofApp(in: Bundle.main.bundleURL.deletingLastPathComponent())

    var body: some Scene {
        Window("AdaMus", id: "main") {
            ProgramView()
                .environment(program)
                .task { program.start() }
        }
        .defaultSize(Metrics.window)
        .commands {
            WorkspaceCommands()
        }
    }
}
