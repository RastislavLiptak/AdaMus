// A new session, in a sheet over the window (HIG "Sheets"): the form, in the order the session
// needs it (the source with its recording, the audio output, the style of its music, the name),
// then the session as it runs, with a button that stops it.
// Once it has ended and been analysed, the sheet closes on its overview.
import SwiftUI

struct NewSessionSheet: View {
    let ended: (String) -> Void
    /// A run that did not end well may still have left a session to list.
    let failed: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var model: NewSessionModel
    @State private var run: RunModel?

    init(api: APIClient, ended: @escaping (String) -> Void, failed: @escaping () -> Void) {
        self.ended = ended
        self.failed = failed
        _model = State(initialValue: NewSessionModel(api: api))
    }

    var body: some View {
        Group {
            if let run {
                RunningView(run: run, name: model.name)
                    .task {
                        await run.follow()
                        if !Task.isCancelled { finish(run) }  // not when the window has gone
                    }
            } else {
                SessionFormView(model: model) { started in run = RunModel(api: model.api, state: started) }
            }
        }
        .frame(width: Metrics.sheetWidth)
        .interactiveDismissDisabled(run != nil)
        .task { await model.open() }
        .alert(
            model.runEnded ? "The Session Did Not Finish" : "The Session Could Not Start",
            isPresented: Binding(get: { model.problem != nil }, set: { if !$0 { model.dismissProblem() } })
        ) {
            Button("OK") {}
        } message: {
            Text(model.problem ?? "")
        }
    }

    private func finish(_ run: RunModel) {
        switch run.outcome {
        case .stored(let id):
            ended(id)
            dismiss()
        case .failed(let why):
            self.run = nil  // back to the form, which says what went wrong
            model.reportRun(problem: why, started: run.state.sessionId != nil)
            failed()
        }
    }
}

struct SessionFormView: View {
    @Bindable var model: NewSessionModel
    let started: (RunState) -> Void

    var body: some View {
        Form {
            Section {
                sourcePicker
                if let recordings = model.source?.recordings, !recordings.isEmpty {
                    Picker(
                        "Recording", selection: Binding(get: { model.recording }, set: { model.chosenRecording = $0 })
                    ) {
                        ForEach(recordings, id: \.self) { Text($0).tag(Optional($0)) }
                    }
                    .accessibilityLabel("Recording")  // as the name below
                }
                outputPicker
                musicPicker
                TextField("Name", text: $model.name, prompt: Text("Required"))
                    .accessibilityLabel("Name")  // the form shows its title beside it, unread
                if let hint = model.nameProblem {
                    Text(hint).font(.callout).foregroundStyle(.red)
                }
            } footer: {
                VStack(alignment: .leading, spacing: Metrics.Spacing.tight) {
                    if let warning = model.check?.warning {
                        Label(warning, systemImage: "exclamationmark.triangle").foregroundStyle(.secondary)
                    }
                    if let problem = model.check?.problem {
                        Label(problem, systemImage: "xmark.octagon").foregroundStyle(.red)
                    }
                }
                .font(.callout)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("New Session")
        // what the program says of the choice; a later choice cancels the look at an earlier one
        .task(id: [model.source?.id, model.recording]) { await model.checkChoice() }
        .onDisappear { model.stopSample() }
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel", role: .cancel) { dismiss() }
            }
            ToolbarItem(placement: .confirmationAction) {
                Button("Start") {
                    Task { if let state = await model.start() { started(state) } }
                }
                .disabled(!model.canStart)
            }
        }
    }

    @Environment(\.dismiss) private var dismiss

    @ViewBuilder private var sourcePicker: some View {
        if let form = model.form, form.sources.isEmpty {
            LabeledContent("Source") {
                Text(model.lookingForHeadset ? "Looking for the headset…" : form.sourcesHint)
                    .foregroundStyle(.secondary)
            }
        } else {
            LabeledContent("Source") {
                HStack {
                    Picker("Source", selection: Binding(get: { model.source?.id }, set: { model.chosenSource = $0 })) {
                        ForEach(model.form?.sources ?? []) { Text($0.title).tag(Optional($0.id)) }
                    }
                    .labelsHidden()
                    lookAgain
                }
            }
        }
    }

    /// The style of the music, and beside it a button that plays a sample of the one chosen or
    /// stops it, as the buttons that look again stand beside the source and the output.
    private var musicPicker: some View {
        LabeledContent("Music") {
            HStack {
                Picker("Music", selection: Binding(get: { model.style?.id }, set: { model.choose(style: $0) })) {
                    ForEach(model.form?.styles ?? []) { Text($0.title).tag(Optional($0.id)) }
                }
                .labelsHidden()
                let playing = model.playingSample
                Button(playing ? "Stop Sample" : "Play Sample", systemImage: playing ? "stop.fill" : "play.fill") {
                    model.playOrStopSample()
                }
                .labelStyle(.iconOnly)
                .help(playing ? "Stop the sample" : "Play a sample of the music chosen")
            }
        }
    }

    @ViewBuilder private var outputPicker: some View {
        LabeledContent("Audio Output") {
            HStack {
                if let outputs = model.outputs, outputs.outputs.isEmpty {
                    Text(outputs.hint).foregroundStyle(.secondary)
                } else {
                    Picker(
                        "Audio Output", selection: Binding(get: { model.output?.id }, set: { model.chosenOutput = $0 })
                    ) {
                        ForEach(model.outputs?.outputs ?? []) { Text($0.title).tag(Optional($0.id)) }
                    }
                    .labelsHidden()
                }
                lookForOutputs
            }
        }
    }

    /// The button that looks for the audio outputs again; while they are read, a spinner in its
    /// place.
    @ViewBuilder private var lookForOutputs: some View {
        if model.lookingForOutputs {
            ProgressView()
                .controlSize(.small)
                .help("Looking for audio outputs")
        } else {
            Button("Look for Audio Outputs Again", systemImage: "arrow.clockwise") {
                Task { await model.lookForOutputs() }
            }
            .labelStyle(.iconOnly)
            .help("Look for audio outputs again")
        }
    }

    /// The button that looks for the headset again; while it looks, a spinner in its place
    /// (HIG "Progress indicators": by the control that began it, without a label).
    @ViewBuilder private var lookAgain: some View {
        if model.lookingForHeadset {
            ProgressView()
                .controlSize(.small)
                .help("Looking for the headset")
        } else {
            Button("Look for the Headset Again", systemImage: "arrow.clockwise") {
                Task { await model.lookForSources() }
            }
            .labelStyle(.iconOnly)
            .help("Look for the headset again")
        }
    }
}

/// The session as it runs: what it is doing, and the button that stops it.
struct RunningView: View {
    let run: RunModel
    let name: String

    /// Stop is no default button: a Return pressed by chance does not end a recording (HIG
    /// "Alerts": no default where people should not act without reading).
    var body: some View {
        VStack(spacing: Metrics.Spacing.regular) {
            ProgressView()
            Text(title).font(.title3.weight(.semibold))
            Text(name).foregroundStyle(.secondary)
            Button("Stop") { Task { await run.stop() } }
                .controlSize(.large)
                .disabled(run.stopping || run.state.phase == .finishing)
                .padding(.top, Metrics.Spacing.small)
            if let problem = run.problem {
                Text(problem).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
        }
        .padding(Metrics.Spacing.sheet)
        .frame(maxWidth: .infinity)
    }

    private var title: String {
        switch run.state.phase {
        case .starting, .idle: "Starting the Session…"
        case .running: run.stopping ? "Stopping the Session…" : "Session Is Running"
        case .finishing: "Analysing the Session…"
        }
    }
}
