// The song library, in its window, which the Window menu opens: a song selected, then changed in
// the editor, a sheet over the window: its title, and its parts, where a stretch dragged over
// clings to the part beside it. The stand-in song is renamed and marked anew, rest then focus,
// then given its title back, so the test starts an app of its own.
import XCTest

final class LibraryTests: WindowTests {
    func testTheLibraryOpensFromTheWindowMenuAndASongIsChangedInTheEditor() {
        app.menuBars.menuBarItems["Window"].click()
        app.menuItems["Song Library"].click()
        let window = app.windows["Song Library"]
        XCTAssertTrue(window.waitForExistence(timeout: patience), "its own window")
        let row = window.cells.containing(.staticText, identifier: "Stand-in Song").firstMatch
        XCTAssertTrue(row.staticTexts["Unknown Artist"].waitForExistence(timeout: patience), "as Apple Music lists it")
        row.click()
        XCTAssertTrue(window.buttons["Play Song"].waitForExistence(timeout: patience), "the song")

        let editor = edit(window)
        let track = editor.descendants(matching: .any)["Song Track"]
        let done = editor.buttons["Done"]
        XCTAssertTrue(done.isEnabled, "marked rest and focus as it is")
        drag(along: track, from: 0.01, to: 0.995)  // its end clings to the end of the song
        let menu = app.popovers.firstMatch
        XCTAssertTrue(menu.buttons["Remove Mark"].waitForExistence(timeout: patience), "a menu over what is selected")
        menu.buttons["Remove Mark"].click()
        XCTAssertTrue(menu.waitForNonExistence(timeout: patience), "the menu closes once a mark is chosen")
        let disabled = expectation(for: NSPredicate(format: "isEnabled == false"), evaluatedWith: done)
        XCTAssertEqual(XCTWaiter.wait(for: [disabled], timeout: patience), .completed, "no focus: it cannot be kept")
        // marked anew: its first half rest, its second focus, whose end dragged near the rest clings to it
        drag(along: track, from: 0.005, to: 0.5)  // beside the track: from its start
        menu.buttons["Rest"].click()
        drag(along: track, from: 0.995, to: 0.505)
        menu.buttons["Focus"].click()
        retitle(editor, "Tokyo")
        XCTAssertTrue(becomesEnabled(done))
        done.click()
        XCTAssertTrue(editor.waitForNonExistence(timeout: patience), "done, the editor closes")
        XCTAssertTrue(window.staticTexts["Tokyo"].waitForExistence(timeout: patience), "the song as it was changed")

        let again = edit(window)
        retitle(again, "Stand-in Song")
        again.buttons["Done"].click()
        XCTAssertTrue(window.cells.staticTexts["Stand-in Song"].waitForExistence(timeout: patience), "as it was")
        window.buttons[XCUIIdentifierCloseWindow].click()
    }

    /// The editor of the song selected, opened from the toolbar of ``window``.
    private func edit(_ window: XCUIElement) -> XCUIElement {
        window.buttons["Edit…"].click()
        let editor = window.sheets.firstMatch
        XCTAssertTrue(editor.textFields["Title"].waitForExistence(timeout: patience), "a sheet over the window")
        return editor
    }

    /// Type ``title`` over the title in ``editor``.
    private func retitle(_ editor: XCUIElement, _ title: String) {
        let field = editor.textFields["Title"]
        field.click()
        field.typeKey("a", modifierFlags: .command)
        field.typeText(title)
    }

    /// Drag along ``element`` from the share ``from`` of its width to ``to``, at its middle height.
    private func drag(along element: XCUIElement, from: CGFloat, to: CGFloat) {
        let start = element.coordinate(withNormalizedOffset: CGVector(dx: from, dy: 0.5))
        start.click(
            forDuration: 0.1, thenDragTo: element.coordinate(withNormalizedOffset: CGVector(dx: to, dy: 0.5)))
    }
}
