// The window as the operator uses it: clicked and typed into, as a person would, against the
// real program on the stand-in computer of the tests, which tests/app/test_app.py starts for
// these tests alone and the app is told of (Program.ofApp). Run alone, from Xcode, they are
// skipped. What the tests share: how the app is started, and the steps of the operator they
// all take. The tests that only look (BrowsingTests) share one app; those that make a session
// (NewSessionTests) each start their own.
import XCTest

@MainActor
class WindowTests: XCTestCase {
    let app = XCUIApplication()
    /// The longest a step of the program takes here: a session over a recording runs to its end
    /// and is analysed in seconds; one with the headset runs until it is stopped
    /// (tests/support/stand_in_program.py).
    let patience: TimeInterval = 30

    /// Whether each test starts the app anew and closes it at its end; if not, the tests of the
    /// class go on in the app the one before left, brought back to the start (``backToTheStart``),
    /// and start it anew only once it has gone, and at the first of them.
    class var startsTheAppAnew: Bool { true }
    /// The classes whose first test has started the app in this run: an app left running before
    /// it, by a run that was cut short, talks to a program that has gone.
    private static var launched: Set<ObjectIdentifier> = []

    override func setUp() async throws {
        let given = ProcessInfo.processInfo.environment
        guard let address = given["ADAMUS_ADDRESS"], let token = given["ADAMUS_TOKEN"] else {
            throw XCTSkip("run by tests/app/test_app.py, which starts the program")
        }
        continueAfterFailure = false
        app.launchEnvironment = ["ADAMUS_PROGRAM_ADDRESS": address, "ADAMUS_PROGRAM_TOKEN": token]
        let first = WindowTests.launched.insert(ObjectIdentifier(Self.self)).inserted
        if Self.startsTheAppAnew || first || [.notRunning, .unknown].contains(app.state) {
            app.launch()  // and the app running before ended
        } else {
            app.activate()
            backToTheStart()
        }
        // listed and laid out: the list is there a moment before its rows can be clicked
        let listed = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "isHittable == true"), object: app.outlines.firstMatch.cells.firstMatch)
        XCTAssertEqual(XCTWaiter.wait(for: [listed], timeout: patience), .completed, "the sessions in the sidebar")
    }

    override func tearDown() async throws {
        if Self.startsTheAppAnew { app.terminate() }
    }

    /// The window as a new app shows it, whatever the test before left in it.
    func backToTheStart() {}

    // MARK: The steps of the operator

    /// The start of a session on its overview: its name over its value, read as one.
    var started: XCUIElement {
        app.descendants(matching: .any).matching(NSPredicate(format: "label BEGINSWITH 'Started'"))
            .firstMatch
    }

    func newSession() -> XCUIElement {
        app.typeKey("n", modifierFlags: .command)
        let sheet = app.sheets.firstMatch
        XCTAssertTrue(sheet.waitForExistence(timeout: patience), "⌘N opens the form")
        XCTAssertTrue(sheet.textFields["Name"].waitForExistence(timeout: patience))
        sheet.textFields["Name"].click()
        return sheet
    }

    func choose(source title: String, in sheet: XCUIElement) {
        sheet.popUpButtons["Source"].click()
        sheet.menuItems[title].click()
        XCTAssertEqual(sheet.popUpButtons["Source"].value as? String, title)
        sheet.textFields["Name"].click()
    }

    /// A text of the form: some say it as their label, some as their value.
    func text(_ said: String, in sheet: XCUIElement) -> XCUIElement {
        sheet.staticTexts.matching(NSPredicate(format: "label == %@ OR value == %@", said, said))
            .firstMatch
    }

    func becomesEnabled(_ element: XCUIElement) -> Bool {
        let enabled = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "isEnabled == true"), object: element)
        return XCTWaiter.wait(for: [enabled], timeout: patience) == .completed
    }

    func sidebarRow(_ name: String) -> XCUIElement {
        app.outlines.firstMatch.cells.containing(.staticText, identifier: name).firstMatch
    }
}
