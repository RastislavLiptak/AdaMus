// A stored session looked at in the window, and a new one begun but cancelled: nothing these
// tests do stays, so they share one app, each test in it from where the app starts.
import XCTest

final class BrowsingTests: WindowTests {
    override class var startsTheAppAnew: Bool { false }
    /// How many things may be open one over another: a popover over a sheet over the window.
    private static let mostOpenAtOnce = 3

    /// Closed what a test before left open, the inspector hidden, no session chosen: chosen
    /// anew, a session is shown as at first (its overview, not played, its log).
    override func backToTheStart() {
        for _ in 0..<Self.mostOpenAtOnce where app.sheets.count + app.popovers.count > 0 {
            app.typeKey(.escape, modifierFlags: [])
        }
        let hide = app.toolbars.buttons["Hide Inspector"]
        if hide.exists {
            app.radioButtons["Timeline"].click()  // the inspector is beside the timeline only
            hide.click()
        }
        let chosen = app.outlines.firstMatch.cells.matching(NSPredicate(format: "isSelected == true"))
            .firstMatch
        if chosen.exists {
            XCUIElement.perform(withKeyModifiers: .command) { chosen.click() }  // let go of
        }
        XCTAssertFalse(chosen.exists, "no session chosen")
    }

    nonisolated override class func tearDown() {
        MainActor.assumeIsolated { XCUIApplication().terminate() }  // the app the tests shared
        super.tearDown()
    }

    // MARK: A new session begun but cancelled

    func testANameAgainstTheRuleIsSaidAndCannotStartAndCancelLeavesNothing() {
        let sheet = newSession()
        chooseSong(in: sheet)
        choose(source: "synthetic", in: sheet)
        sheet.textFields["Name"].typeText("a/b")
        XCTAssertTrue(text("Use at most 50 characters, without “/” or “:”.", in: sheet).exists)
        XCTAssertFalse(sheet.buttons["Start"].isEnabled)
        sheet.textFields["Name"].typeKey(.delete, modifierFlags: [])
        sheet.textFields["Name"].typeKey(.delete, modifierFlags: [])
        XCTAssertTrue(sheet.buttons["Start"].isEnabled, "“a” keeps the rule")
        let sessions = app.outlines.firstMatch.cells.count
        sheet.buttons["Cancel"].click()
        XCTAssertTrue(sheet.waitForNonExistence(timeout: patience))
        XCTAssertEqual(app.outlines.firstMatch.cells.count, sessions)
    }

    // MARK: A stored session

    func testTheOverviewTheTimelineAndTheDetailsAreSwitchedInTheToolbar() {
        sidebarRow("morning_focus").click()
        XCTAssertTrue(started.waitForExistence(timeout: patience), "the overview first")
        let zoom = app.toolbars.buttons["Zoom In"]
        XCTAssertFalse(zoom.isEnabled, "the tools of the timeline are there, but off")
        app.radioButtons["Timeline"].click()
        XCTAssertTrue(becomesEnabled(zoom), "the timeline, with its zoom")
        XCTAssertFalse(started.exists)
        app.radioButtons["Details"].click()
        XCTAssertTrue(
            app.staticTexts["Configuration"].waitForExistence(timeout: patience), "the details")
        XCTAssertFalse(zoom.isEnabled)
        app.radioButtons["Overview"].click()
        XCTAssertTrue(started.waitForExistence(timeout: patience))
    }

    func testALogOpenedAfterItsLinesCameShowsItsNewestLine() throws {
        sidebarRow("morning_focus").click()
        app.radioButtons["Timeline"].click()
        let show = app.toolbars.buttons["Show Inspector"]
        XCTAssertTrue(becomesEnabled(show), "the timeline, with its logs beside it")
        app.buttons["Play"].click()
        sleep(12)  // the lines of the log come as it plays, more than the inspector shows at once
        app.buttons["Pause"].click()
        show.click()
        try showsTheNewestLines("opened")
        app.radioButtons["Events"].click()
        app.radioButtons["Log"].click()  // the log opened anew, all its lines at once
        try showsTheNewestLines("chosen again")
    }

    /// That the inspector shows the newest lines of its log, the oldest out of view above.
    private func showsTheNewestLines(_ when: String) throws {
        XCTAssertNotNil(firstLineOfTheInspector(), "the lines of the log, \(when)")
        let (lines, playhead) = try timesShown()
        let shown = lines.compactMap(\.seconds)
        let paused = try XCTUnwrap(playhead?.seconds, "the time of the playhead")
        XCTAssertGreaterThan(shown.min() ?? 0, 0, "the oldest lines are above, out of view, \(when)")
        XCTAssertGreaterThan(
            shown.max() ?? 0, paused - 1, "the newest lines in view, \(when): \(shown)")
    }

    /// A time the window shows, as the computer's language writes it (its tenths after a point or
    /// a comma), and where it is on the screen.
    private struct ShownTime {
        let said: String
        let frame: CGRect

        /// The seconds it stands for: −1:02,5 or 1:02.5.
        var seconds: Double? {
            let negative = said.hasPrefix("−") || said.hasPrefix("-")
            let parts = said.trimmingCharacters(in: CharacterSet(charactersIn: "−-"))
                .replacingOccurrences(of: ",", with: ".").split(separator: ":")
            guard parts.count == 2, let minutes = Double(parts[0]), let rest = Double(parts[1]) else {
                return nil
            }
            return (negative ? -1 : 1) * (minutes * 60 + rest)
        }
    }

    /// The times the window shows, read in one look at the whole of it, not one by one, which
    /// takes seconds each and lets the window change between them: those of the lines of the
    /// inspector at its right, top to bottom, where its scroll area shows them, and the time of
    /// the playhead, the first at its left.
    private func timesShown() throws -> (lines: [ShownTime], playhead: ShownTime?) {
        let window = try app.windows.firstMatch.snapshot()
        var times = [ShownTime]()
        var areas = [CGRect]()
        var left = [window]
        while let next = left.popLast() {
            if next.elementType == .scrollView { areas.append(next.frame) }
            if next.elementType == .staticText, let said = next.value as? String,
                said.wholeMatch(of: #/.*\d:\d\d[.,]\d/#) != nil
            {
                times.append(ShownTime(said: said, frame: next.frame))
            }
            left += next.children
        }
        let middle = window.frame.midX
        let inView = { (time: ShownTime) in
            areas.contains {
                $0.minX > middle && $0.contains(CGPoint(x: time.frame.midX, y: time.frame.midY))
            }
        }
        let lines = times.filter { $0.frame.minX > middle && inView($0) }.sorted {
            $0.frame.minY < $1.frame.minY
        }
        return (lines, times.filter { $0.frame.maxX < middle }.min { $0.frame.minX < $1.frame.minX })
    }

    /// The time of the first line of the log the inspector shows, once there is one.
    private func firstLineOfTheInspector() -> ShownTime? {
        let deadline = Date().addingTimeInterval(patience)
        while Date() < deadline {
            if let found = try? timesShown().lines.first { return found }
            sleep(1)
        }
        return nil
    }

    /// It copies into the clipboard of the computer, as the operator's copy would.
    func testALineOfALogIsCopiedFromItsMenu() throws {
        sidebarRow("morning_focus").click()
        app.radioButtons["Timeline"].click()
        let show = app.toolbars.buttons["Show Inspector"]
        XCTAssertTrue(becomesEnabled(show), "the timeline, with its logs beside it")
        show.click()
        app.radioButtons["Events"].click()
        app.buttons["Play"].click()  // played, so that its events come
        XCTAssertNotNil(firstLineOfTheInspector(), "a line of the events")
        app.buttons["Pause"].click()
        let time = try XCTUnwrap(
            firstLineOfTheInspector(), "a line of the events, by its time, where it stays")
        NSPasteboard.general.clearContents()
        let line = CGPoint(x: time.frame.midX, y: time.frame.midY)
        onTheScreen(line).rightClick()
        let copy = try XCTUnwrap(openMenuItem("Copy", by: line), "Copy in the menu of the line")
        onTheScreen(copy).click()
        let copied = NSPasteboard.general.string(forType: .string) ?? ""
        XCTAssertTrue(copied.hasPrefix(time.said + "\t"), "the whole line: \(copied)")
    }

    /// A point of the screen, to be clicked there.
    private func onTheScreen(_ point: CGPoint) -> XCUICoordinate {
        let window = app.windows.firstMatch
        return window.coordinate(withNormalizedOffset: .zero)
            .withOffset(CGVector(dx: point.x - window.frame.minX, dy: point.y - window.frame.minY))
    }

    /// The middle of the item ``title`` of the menu opened by ``point``, read in one look at the
    /// whole app: asked one by one, every item of every menu of the app, the menu bar's too,
    /// took seconds each. The closed menus' items lie away from the point, if anywhere.
    private func openMenuItem(_ title: String, by point: CGPoint) throws -> CGPoint? {
        var items = [CGRect]()
        var left = [try app.snapshot()]
        while let next = left.popLast() {
            if next.elementType == .menuItem, next.title == title, !next.frame.isEmpty { items.append(next.frame) }
            left += next.children
        }
        let away = { (item: CGRect) in hypot(item.midX - point.x, item.midY - point.y) }
        return items.min { away($0) < away($1) }.map { CGPoint(x: $0.midX, y: $0.midY) }
    }

    /// The playhead put where a click in the lanes puts it, then moved on by a drag along them:
    /// played, the session goes on from there, not from its start.
    func testThePlayheadPutByAClickAndMovedByADragPlaysFromThere() throws {
        sidebarRow("morning_focus").click()
        app.radioButtons["Timeline"].click()
        XCTAssertTrue(becomesEnabled(app.toolbars.buttons["Zoom In"]), "the timeline")
        let detection = app.staticTexts["Detection"]  // the name of its track, left of its lane
        XCTAssertTrue(detection.waitForExistence(timeout: patience))
        let lane = detection.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let first = try playhead()
        lane.withOffset(CGVector(dx: 250, dy: 0)).click()
        let clicked = try playhead()
        XCTAssertGreaterThan(clicked, first + 1, "put further on by the click")
        lane.withOffset(CGVector(dx: 250, dy: 0))
            .click(forDuration: 0.1, thenDragTo: lane.withOffset(CGVector(dx: 400, dy: 0)))
        let dragged = try playhead()
        XCTAssertGreaterThan(dragged, clicked + 1, "and on by the drag")
        app.buttons["Play"].click()
        sleep(3)
        app.buttons["Pause"].click()
        let played = try playhead()
        XCTAssertGreaterThan(played, dragged + 0.5, "played on")
        XCTAssertLessThan(played, dragged + 5, "from where it was put, not from the start: \(dragged) → \(played)")
    }

    /// The seconds the playhead is at.
    private func playhead() throws -> Double {
        try XCTUnwrap(timesShown().playhead?.seconds, "the time of the playhead")
    }

    /// A part marked with Shift and a drag is cut out of the data of the session and copied: into
    /// the clipboard of the computer, as the operator's copy would be.
    func testAPartMarkedWithShiftAndADragIsCopiedAsItsData() throws {
        sidebarRow("morning_focus").click()
        app.radioButtons["Timeline"].click()
        XCTAssertTrue(becomesEnabled(app.toolbars.buttons["Zoom In"]), "the timeline")
        let detection = app.staticTexts["Detection"]  // the name of its track, left of its lane
        XCTAssertTrue(detection.waitForExistence(timeout: patience))
        let from = detection.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            .withOffset(CGVector(dx: 200, dy: 0))  // in the lane
        XCUIElement.perform(withKeyModifiers: .shift) {
            from.click(forDuration: 0.1, thenDragTo: from.withOffset(CGVector(dx: 200, dy: 0)))
        }
        let sheet = app.sheets.firstMatch
        XCTAssertTrue(sheet.waitForExistence(timeout: patience), "the sheet of the part marked")
        let copy = sheet.buttons["Copy"]
        XCTAssertTrue(becomesEnabled(copy), "once the program has cut the part out")
        NSPasteboard.general.clearContents()
        copy.click()
        XCTAssertTrue(sheet.waitForNonExistence(timeout: patience))
        let copied = NSPasteboard.general.string(forType: .string) ?? ""
        let data = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(copied.utf8)) as? [String: Any],
            "the part, as data: \(copied.prefix(300))")
        XCTAssertEqual((data["session"] as? [String: Any])?["name"] as? String, "morning_focus")
        XCTAssertNotNil(data["detector"] as? [String: Any])
    }

    // MARK: The electrodes on the head

    func testAnElectrodeChosenOnTheHeadOfTheOverviewSaysWhatItRecords() throws {
        sidebarRow("morning_focus").click()
        XCTAssertTrue(started.waitForExistence(timeout: patience), "the overview")
        let placeholder = text("Click an electrode to see what it records.", in: app.windows.firstMatch)
        XCTAssertTrue(placeholder.waitForExistence(timeout: patience), "the head, nothing chosen yet")
        keep("the head of the overview")
        app.buttons["AF7"].click()  // on its dot, where VoiceOver finds it too
        let about = app.staticTexts.matching(
            NSPredicate(format: "value BEGINSWITH 'AF7 records the left prefrontal'")
        )
        .firstMatch
        XCTAssertTrue(
            about.waitForExistence(timeout: patience), "what it records, once the head has turned")
        XCTAssertFalse(placeholder.exists)
        keep("AF7 chosen")
        app.buttons["Fpz"].click()  // the reference of the Muse
        XCTAssertTrue(
            app.staticTexts.matching(NSPredicate(format: "value BEGINSWITH 'Fpz is the reference'"))
                .firstMatch
                .waitForExistence(timeout: patience))
        keep("the reference chosen")
    }

    func testTheNameOfAnElectrodesTrackOpensWhereItIsOnTheHead() throws {
        sidebarRow("morning_focus").click()
        app.radioButtons["Timeline"].click()
        XCTAssertTrue(becomesEnabled(app.toolbars.buttons["Zoom In"]), "the timeline")
        let name = app.buttons["TP9"]
        XCTAssertTrue(name.waitForExistence(timeout: patience), "the name of the track, a button")
        name.click()
        let popover = app.popovers.firstMatch
        XCTAssertTrue(popover.waitForExistence(timeout: patience), "the popover of TP9")
        XCTAssertTrue(
            popover.staticTexts.matching(
                NSPredicate(format: "value BEGINSWITH 'TP9 records where the left temporal'")
            )
            .firstMatch.exists)
        keep("the popover of TP9")
        app.typeKey(.escape, modifierFlags: [])
        XCTAssertTrue(popover.waitForNonExistence(timeout: patience), "closed")
    }

    /// A picture of the window, kept with the results, to be looked at.
    private func keep(_ name: String) {
        let picture = XCTAttachment(screenshot: app.windows.firstMatch.screenshot())
        picture.name = name
        picture.lifetime = .keepAlways
        add(picture)
    }

    func testTheNameInTheToolbarRenamesTheSessionOnceTheFocusLeavesItOrReturnIsPressed() {
        sidebarRow("morning_focus").click()
        XCTAssertTrue(started.waitForExistence(timeout: patience))
        app.radioButtons["Details"].click()
        let name = app.toolbars.textFields["Session Name"]
        rename(name, to: "a/b", then: .tab)
        let refused = app.staticTexts["Use at most 50 characters, without “/” or “:”."]
        XCTAssertTrue(refused.waitForExistence(timeout: patience), "a name no folder can have is said")
        app.buttons["OK"].click()
        XCTAssertEqual(name.value as? String, "morning_focus", "and the name stays")

        rename(name, to: "evening_focus", then: .tab)
        XCTAssertTrue(sidebarRow("evening_focus").waitForExistence(timeout: patience), "renamed once the focus left")
        XCTAssertFalse(sidebarRow("morning_focus").exists)
        XCTAssertTrue(sidebarRow("evening_focus").isSelected, "still chosen")
        // a segment says it is chosen by its value
        XCTAssertEqual(app.radioButtons["Details"].value as? Int, 1, "on the same part")
        keep("the toolbar with the new name")
        app.radioButtons["Overview"].click()
        XCTAssertTrue(started.waitForExistence(timeout: patience), "read under its new name")

        rename(app.toolbars.textFields["Session Name"], to: "morning_focus", then: .return)
        XCTAssertTrue(sidebarRow("morning_focus").waitForExistence(timeout: patience), "renamed with Return")
    }

    private func rename(_ field: XCUIElement, to name: String, then key: XCUIKeyboardKey) {
        XCTAssertTrue(field.waitForExistence(timeout: patience))
        field.click()
        field.typeKey("a", modifierFlags: .command)
        field.typeText(name)
        field.typeKey(key, modifierFlags: [])
    }

    func testADeletionIsConfirmedFirstAndCanBeCancelled() {
        sidebarRow("morning_focus").click()
        app.typeKey(.delete, modifierFlags: [])
        let question = app.staticTexts["Delete “morning_focus”?"]
        XCTAssertTrue(question.waitForExistence(timeout: patience))
        app.buttons["Cancel"].click()
        XCTAssertTrue(question.waitForNonExistence(timeout: patience))
        XCTAssertTrue(sidebarRow("morning_focus").exists, "nothing is deleted unasked")
    }

    func testTheToolbarDeletesTheChosenSessionAfterTheSameQuestion() {
        sidebarRow("morning_focus").click()
        keep("the toolbar with its delete button")
        app.toolbars.buttons["Delete…"].click()
        let question = app.staticTexts["Delete “morning_focus”?"]
        XCTAssertTrue(question.waitForExistence(timeout: patience))
        app.buttons["Cancel"].click()
        XCTAssertTrue(question.waitForNonExistence(timeout: patience))
        XCTAssertTrue(sidebarRow("morning_focus").exists, "nothing is deleted unasked")
    }
}
