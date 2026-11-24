// The window once the program is ready: the sessions in a sidebar, the one chosen beside it
// (HIG "Sidebars", "Split views"). A session is deleted from the sidebar, as a file in Finder:
// with its context menu, Edit ▸ Delete or the Delete key, or with the toolbar's button, then
// confirmed, as it cannot be undone (HIG "Alerts").
import SwiftUI

struct WorkspaceView: View {
    @State private var workspace: Workspace

    init(api: APIClient) {
        _workspace = State(initialValue: Workspace(api: api))
    }

    var body: some View {
        NavigationSplitView {
            SessionList(workspace: workspace)
                .navigationSplitViewColumnWidth(
                    min: Metrics.sidebar.min, ideal: Metrics.sidebar.ideal, max: Metrics.sidebar.max)
        } detail: {
            if let session = workspace.session {
                SessionView(session: session, showsInspector: $workspace.showsInspector) {
                    workspace.askToDelete(session.summary.id)
                }
                .id(session.summary.id)
            } else if let problem = workspace.listProblem, workspace.sessions.isEmpty {
                ContentUnavailableView {
                    Label("The Sessions Could Not Be Read", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(problem)
                } actions: {
                    Button("Try Again") { Task { await workspace.load() } }
                }
                .navigationTitle("Sessions")
            } else if workspace.sessions.isEmpty {
                ContentUnavailableView {
                    Label("No Sessions", systemImage: "waveform.path.ecg")
                } description: {
                    Text("A session you run appears here.")
                } actions: {
                    Button("New Session") { workspace.makeSession() }
                }
                .navigationTitle("Sessions")
            } else {
                ContentUnavailableView(
                    "No Session Selected", systemImage: "waveform.path.ecg",
                    description: Text("Choose a session in the sidebar.")
                )
                .navigationTitle("Sessions")
            }
        }
        .sheet(isPresented: $workspace.isMakingSession) {
            NewSessionSheet(api: workspace.api) { id in
                Task { await workspace.sessionEnded(id) }
            } failed: {
                Task { await workspace.load() }
            }
        }
        .alert(
            "Delete “\(workspace.toDelete?.name ?? "")”?",
            isPresented: Binding(get: { workspace.toDelete != nil }, set: { if !$0 { workspace.toDelete = nil } }),
            presenting: workspace.toDelete
        ) { session in
            // no default: a Return pressed by chance deletes nothing (HIG "Alerts")
            Button("Delete", role: .destructive) { Task { await workspace.delete(session) } }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("The session and everything recorded in it are deleted. You can’t undo this action.")
        }
        .alert(
            "The Request Could Not Be Done",
            isPresented: Binding(get: { workspace.problem != nil }, set: { if !$0 { workspace.problem = nil } })
        ) {
            Button("OK") {}
        } message: {
            Text(workspace.problem ?? "")
        }
        .focusedSceneValue(\.workspace, workspace)
        .onDisappear { workspace.session?.stop() }  // the program has ended: no music plays on
        .task { await workspace.load() }
    }
}

struct SessionList: View {
    @Bindable var workspace: Workspace

    var body: some View {
        List(selection: $workspace.selection) {
            if let problem = workspace.listProblem {
                Label(problem, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
            }
            ForEach(workspace.sessions) { session in
                Text(session.name)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(session.name)
                    .tag(session.id)
            }
        }
        .contextMenu(forSelectionType: SessionSummary.ID.self) { chosen in
            Button("Delete…") { workspace.askToDelete(chosen.first) }
                .disabled(chosen.count != 1)
        }
        .onDeleteCommand { workspace.askToDelete(workspace.selection) }
        // at the foot of the sidebar, as New Folder in Notes: in its toolbar, hidden with the
        // sidebar, the button made the toolbar lay itself out again without end, at a whole core
        .safeAreaInset(edge: .bottom) {
            Button("New Session", systemImage: "plus") { workspace.makeSession() }
                .buttonStyle(.borderless)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(Metrics.Spacing.regular)
        }
    }
}
