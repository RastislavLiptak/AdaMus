// AdaMus: the main window with the sessions in a sidebar and the one chosen beside it; and the
// window of the song library, which opens only when it is asked for: from the Window menu, which
// lists it, or from the form of a new session. The app quits as its last window closes, and the
// program with it (HIG "Windows").
import SwiftUI

@main
struct AdaMusApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
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
        Window("Song Library", id: LibraryWindow.id) {
            LibraryWindow()
                .environment(program)
        }
        .defaultSize(Metrics.library)
        .defaultLaunchBehavior(.suppressed)  // opened only when asked for, never as the app starts
        .restorationBehavior(.disabled)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
