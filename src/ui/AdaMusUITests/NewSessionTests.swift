// A new session made in the window, from a recording and with the headset: each test in an app
// of its own, as each leaves a session behind it until it has deleted it again.
import XCTest

final class NewSessionTests: WindowTests {
    func testANewSessionFromARecordingRunsAndTheSheetClosesOnItsOverview() {
        let name = "Recording \(UUID().uuidString.prefix(4))"
        let sheet = newSession()
        choose(source: "synthetic", in: sheet)
        XCTAssertEqual(sheet.popUpButtons["Recording"].value as? String, "short")  // its first, chosen
        sheet.textFields["Name"].typeText(name)
        sheet.buttons["Start"].click()
        XCTAssertTrue(
            sheet.waitForNonExistence(timeout: patience), "the sheet closes once the session is stored")
        XCTAssertTrue(sidebarRow(name).waitForExistence(timeout: patience), "listed in the sidebar")
        XCTAssertTrue(sidebarRow(name).isSelected, "and chosen")
        XCTAssertTrue(started.waitForExistence(timeout: patience), "its overview")
        delete(name)
    }

    func testTheStyleIsChosenFromAMenuAndTheOverviewOfTheSessionPlaysIt() {
        let name = "Style \(UUID().uuidString.prefix(4))"
        let sheet = newSession()
        let music = sheet.popUpButtons["Music"]
        XCTAssertTrue(music.waitForExistence(timeout: patience))
        XCTAssertEqual(music.value as? String, "Folk", "the first, chosen")
        music.click()
        XCTAssertEqual(
            ["Folk", "Rock", "Country", "Koto and Taiko", "Oud and Darbuka"].filter { sheet.menuItems[$0].exists }
                .count,
            5)
        sheet.menuItems["Rock"].click()
        XCTAssertEqual(music.value as? String, "Rock")
        XCTAssertTrue(sheet.buttons["Stop Sample"].waitForExistence(timeout: patience), "a sample of it plays")
        sheet.buttons["Stop Sample"].click()
        XCTAssertTrue(sheet.buttons["Play Sample"].waitForExistence(timeout: patience), "and is stopped")
        sheet.buttons["Play Sample"].click()
        XCTAssertTrue(sheet.buttons["Stop Sample"].waitForExistence(timeout: patience), "and played again")
        sheet.buttons["Stop Sample"].click()
        choose(source: "synthetic", in: sheet)
        sheet.textFields["Name"].typeText(name)
        sheet.buttons["Start"].click()
        XCTAssertTrue(sheet.waitForNonExistence(timeout: patience), "the session ran and was stored")
        XCTAssertTrue(started.waitForExistence(timeout: patience), "its overview")
        XCTAssertTrue(app.staticTexts["At Rest"].waitForExistence(timeout: patience), "has the music it played")
        XCTAssertTrue(app.staticTexts["Rock"].exists, "by its name")
        app.buttons["Play Sample"].click()  // on its cover
        XCTAssertTrue(app.buttons["Pause Sample"].waitForExistence(timeout: patience), "the sample it kept plays")
        sleep(1)  // while it plays
        app.buttons["Pause Sample"].click()
        XCTAssertTrue(app.buttons["Play Sample"].exists, "and is paused at once, not once it has played to its end")
        // the time left, under its track
        let clock = "−[0-9]+:[0-9]{2}"
        let left = app.staticTexts.matching(
            NSPredicate(format: "label MATCHES %@ OR value MATCHES %@", clock, clock)
        ).firstMatch
        let shown = { left.label.isEmpty ? left.value as? String : left.label }  // as text() reads
        let paused = shown()
        XCTAssertNotEqual(paused, "−0:20", "it played a while")
        sleep(1)
        XCTAssertEqual(shown(), paused, "and stays where it was paused")
        app.descendants(matching: .any)["Sample"].coordinate(withNormalizedOffset: CGVector(dx: 0.75, dy: 0.5))
            .click()
        XCTAssertEqual(shown(), "−0:05", "a click on its track puts it there, paused")
        app.buttons["Play Sample"].click()
        XCTAssertTrue(app.buttons["Pause Sample"].waitForExistence(timeout: patience))
        app.radioButtons["Timeline"].click()
        app.radioButtons["Overview"].click()
        XCTAssertTrue(app.buttons["Play Sample"].exists, "the overview left, its sample was paused")
        delete(name)
    }

    func testASessionWithTheHeadsetRunsUntilItIsStopped() {
        let name = "Headset \(UUID().uuidString.prefix(4))"
        let sheet = newSession()
        XCTAssertTrue(
            sheet.buttons["Look for the Headset Again"].waitForExistence(timeout: patience),
            "the headset is found: the spinner gives its place back to the button")
        // found, and first
        XCTAssertEqual(sheet.popUpButtons["Source"].value as? String, "Muse 2 (headset)")
        XCTAssertFalse(sheet.popUpButtons["Recording"].exists, "the headset has no recordings")
        sheet.textFields["Name"].typeText(name)
        sheet.buttons["Start"].click()
        let stop = sheet.buttons["Stop"]
        XCTAssertTrue(text("Session Is Running", in: sheet).waitForExistence(timeout: patience))
        stop.click()
        XCTAssertTrue(
            sheet.waitForNonExistence(timeout: patience), "the sheet closes once the session is stored")
        XCTAssertTrue(sidebarRow(name).waitForExistence(timeout: patience))
        delete(name)
    }

    /// Delete a session from the sidebar, as the operator does: the Delete key, then confirmed.
    private func delete(_ name: String) {
        sidebarRow(name).click()
        app.typeKey(.delete, modifierFlags: [])
        XCTAssertTrue(app.staticTexts["Delete “\(name)”?"].waitForExistence(timeout: patience))
        app.buttons["Delete"].click()
        XCTAssertTrue(sidebarRow(name).waitForNonExistence(timeout: patience), "gone from the sidebar")
    }
}
