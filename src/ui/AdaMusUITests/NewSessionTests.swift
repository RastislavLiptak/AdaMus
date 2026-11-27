// A new session made in the window, from a recording and with the headset: each test in an app
// of its own, as each leaves a session behind it until it has deleted it again.
import XCTest

final class NewSessionTests: WindowTests {
    func testANewSessionFromARecordingRunsAndTheSheetClosesOnItsOverview() {
        let name = "Recording \(UUID().uuidString.prefix(4))"
        let sheet = newSession()
        chooseSong(in: sheet)
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

    func testTheSongIsChosenInTheLibraryAndTheOverviewOfTheSessionPlaysTheSongItKept() {
        let name = "Song \(UUID().uuidString.prefix(4))"
        let sheet = newSession()
        let chosen = text("Stand-in Song – Unknown Artist", in: sheet)
        XCTAssertFalse(chosen.exists, "none, until one is chosen")
        XCTAssertFalse(sheet.buttons["Start"].isEnabled)
        sheet.buttons["Choose…"].click()
        // the library, in its own window beside the sheet: Choose at its foot once a song is selected
        let library = app.windows["Song Library"]
        XCTAssertTrue(library.waitForExistence(timeout: patience), "the library")
        XCTAssertFalse(library.buttons["Choose"].exists, "nothing selected, nothing to choose")
        library.cells.containing(.staticText, identifier: "Stand-in Song").firstMatch.click()
        library.buttons["Play Song"].click()
        XCTAssertTrue(library.buttons["Pause Song"].waitForExistence(timeout: patience), "it plays")
        library.buttons["Choose"].click()
        XCTAssertTrue(library.waitForNonExistence(timeout: patience), "chosen, the library closes")
        XCTAssertTrue(chosen.waitForExistence(timeout: patience), "the song chosen")
        choose(source: "synthetic", in: sheet)
        sheet.textFields["Name"].typeText(name)
        sheet.buttons["Start"].click()
        XCTAssertTrue(sheet.waitForNonExistence(timeout: patience), "the session ran and was stored")
        XCTAssertTrue(started.waitForExistence(timeout: patience), "its overview")
        XCTAssertTrue(app.staticTexts["Stand-in Song"].waitForExistence(timeout: patience), "has the song it played")
        XCTAssertTrue(app.staticTexts["Unknown Artist"].exists, "by whom")
        XCTAssertTrue(app.staticTexts["Focus"].exists, "and its parts")
        XCTAssertFalse(app.staticTexts["Song Length"].exists, "what it kept of its song is in its details")
        // the time left, under its track
        let clock = "−[0-9]+:[0-9]{2}"
        let left = app.staticTexts.matching(
            NSPredicate(format: "label MATCHES %@ OR value MATCHES %@", clock, clock)
        ).firstMatch
        let shown = { (left.label.isEmpty ? left.value as? String : left.label) ?? "" }  // as text() reads
        let length = seconds(shown())
        app.buttons["Play Song"].click()  // on its cover
        XCTAssertTrue(app.buttons["Pause Song"].waitForExistence(timeout: patience), "the song it kept plays")
        sleep(1)  // while it plays
        app.buttons["Pause Song"].click()
        XCTAssertTrue(app.buttons["Play Song"].exists, "and is paused at once, not once it has played to its end")
        let paused = shown()
        XCTAssertLessThan(seconds(paused), length, "it played a while")
        sleep(1)
        XCTAssertEqual(shown(), paused, "and stays where it was paused")
        app.descendants(matching: .any)["Song Track"].coordinate(withNormalizedOffset: CGVector(dx: 0.75, dy: 0.5))
            .click()
        XCTAssertEqual(seconds(shown()), length / 4, accuracy: 1, "a click on its track puts it there, paused")
        app.buttons["Play Song"].click()
        XCTAssertTrue(app.buttons["Pause Song"].waitForExistence(timeout: patience))
        app.radioButtons["Timeline"].click()
        app.radioButtons["Overview"].click()
        XCTAssertTrue(app.buttons["Play Song"].exists, "the overview left, its song was paused")
        delete(name)
    }

    func testASessionWithTheHeadsetRunsUntilItIsStopped() {
        let name = "Headset \(UUID().uuidString.prefix(4))"
        let sheet = newSession()
        chooseSong(in: sheet)
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

    /// The seconds of a time left, as “−1:05” says it.
    private func seconds(_ left: String) -> Double {
        let parts = left.dropFirst().split(separator: ":").compactMap { Double($0) }
        return parts.reduce(0) { $0 * 60 + $1 }
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
