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
