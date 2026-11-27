// The commands of the menu bar that are the app's own: a new session, in place of File ▸ New;
// and none that hides the sidebar, which every window of the app always shows.
import SwiftUI

extension FocusedValues {
    @Entry var workspace: Workspace?
}

struct WorkspaceCommands: Commands {
    @FocusedValue(\.workspace) private var workspace

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Session…") { workspace?.makeSession() }
                .keyboardShortcut("n")
                .disabled(workspace?.isInFront != true)
        }
        CommandGroup(replacing: .sidebar) {}
    }
}
