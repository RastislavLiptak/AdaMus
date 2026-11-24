// The detail of a session: its name as the title of the window (when, how long and on what it
// ran are first on its overview), its overview, timeline or details switched by a segmented
// control in the toolbar (HIG "Segmented controls": to switch views in a toolbar); beside the
// timeline, its logs in the inspector. The toolbar also deletes the session, after the same
// question as the sidebar asks. A part once shown stays
// built and is only hidden: on a recorded session, that took a third off the work of a switch.
import SwiftUI

struct SessionView: View {
    @Bindable var session: SessionModel
    @Binding var showsInspector: Bool
    /// Asks whether to delete the session.
    let askToDelete: () -> Void

    var body: some View {
        ZStack {
            ForEach(SessionPart.allCases.filter(session.opened.contains)) { part in
                let shown = part == session.part
                content(of: part).opacity(shown ? 1 : 0).allowsHitTesting(shown).accessibilityHidden(!shown)
            }
        }
        .inspector(isPresented: inspectorShown) {
            if let timeline = session.timeline {
                ConsolesInspector(timeline: timeline)
                    .inspectorColumnWidth(
                        min: Metrics.inspector.min, ideal: Metrics.inspector.ideal, max: Metrics.inspector.max)
            }
        }
        .navigationTitle(session.summary.name)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("View", selection: $session.part) {
                    ForEach(SessionPart.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            ToolbarItem {
                Button("Delete…", systemImage: "trash", action: askToDelete)
                    .foregroundStyle(.red)
                    .help("Delete This Session")
            }
            ToolbarSpacer(.fixed)  // apart from the tools that only look at the session
            timelineTools
        }
    }

    /// The logs beside the timeline only.
    private var inspectorShown: Binding<Bool> {
        Binding(get: { showsInspector && session.part == .timeline }, set: { showsInspector = $0 })
    }

    @ViewBuilder private func content(of part: SessionPart) -> some View {
        switch part {
        case .overview:
            ofTheDashboard { OverviewView(dashboard: $0, electrodes: session.electrodes) }
        case .details:
            ofTheDashboard { DetailsView(dashboard: $0) }
        case .timeline:
            if let timeline = session.timeline, session.timelineProblem == nil {
                TimelineView(model: timeline, electrodes: session.electrodes)
                    .sheet(item: Binding(get: { timeline.asked }, set: { if $0 == nil { timeline.forgetMarked() } })) {
                        part in
                        MarkedPartSheet(part: part, session: session.summary.name) {
                            [api = session.api, id = session.summary.id] in
                            try await api.excerpt(of: id, from: part.startS, to: part.endS).text
                        }
                    }
            } else if let problem = session.timelineProblem {
                unreadable(problem)
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    /// The tools of the timeline, there also beside the dashboard but off: a toolbar that
    /// changed with each switch took twice as long to switch.
    @ToolbarContentBuilder private var timelineTools: some ToolbarContent {
        let timeline = session.part == .timeline ? session.timeline : nil
        ToolbarItemGroup {
            Button("Zoom Out", systemImage: "minus.magnifyingglass") {
                timeline?.zoom(by: 1 / TimelineModel.zoomStep)
            }
            .disabled(timeline?.canZoomOut != true)
            Button("Zoom In", systemImage: "plus.magnifyingglass") {
                timeline?.zoom(by: TimelineModel.zoomStep)
            }
            .disabled(timeline?.canZoomIn != true)
            Button(showsInspector ? "Hide Inspector" : "Show Inspector", systemImage: "sidebar.trailing") {
                showsInspector.toggle()
            }
            .help(showsInspector ? "Hide Inspector" : "Show Inspector")
            .disabled(timeline == nil)
        }
    }

    /// A part drawn from the dashboard the program describes, once it has been read.
    @ViewBuilder private func ofTheDashboard(_ content: (Dashboard) -> some View) -> some View {
        if let dashboard = session.dashboard {
            content(dashboard)
        } else if let problem = session.dashboardProblem {
            unreadable(problem)
        } else {
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func unreadable(_ problem: String) -> some View {
        ContentUnavailableView(
            "This Session Cannot Be Read", systemImage: "exclamationmark.triangle", description: Text(problem))
    }
}
