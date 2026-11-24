// The command of the menu bar that is the app's own: a new session, in place of File ▸ New.
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
    }
}
