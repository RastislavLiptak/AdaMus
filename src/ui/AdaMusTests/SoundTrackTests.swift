// What the sound track of a song or of a session is drawn from: the part each moment is in, and
// what each colour means.
import Testing

struct SoundTrackTests {
    private let song = Song(
        id: "s", title: "S", artist: nil, lengthS: 30, loudness: [1],
        parts: [
            SongPart(startS: 2, endS: 10, mark: "REST", title: "Rest", colour: .indigo),
            SongPart(startS: 10, endS: 20, mark: "FOCUS", title: "Focus", colour: .orange),
            SongPart(startS: 24, endS: 30, mark: "REST", title: "Rest", colour: .indigo),
        ],
        cover: nil)

    @Test func aMomentIsInThePartThatBeginsAtOrBeforeItAndEndsAfterIt() {
        #expect(song.part(at: 1.9) == nil)  // before the first beat: in none
        #expect(song.part(at: 2)?.mark == "REST")
        #expect(song.part(at: 10)?.mark == "FOCUS")  // where one ends, the next begins
        #expect(song.part(at: 22) == nil)  // between two parts
        #expect(song.part(at: 30) == nil)  // the end of the last is past it
    }

    @Test func eachKindOfPartIsMeantOnceInTheOrderItFirstComes() {
        #expect(song.kinds.map(\.title) == ["Rest", "Focus"])
    }

    @Test func theEndOfADragClingsToTheEdgeOfAPartOrOfTheSoundNearIt() {
        #expect(song.clinging(9.7, within: 0.5) == 10)  // where the next part begins
        #expect(song.clinging(21, within: 0.5) == 21)  // nothing near
        #expect(song.clinging(20.4, within: 0.5) == 20)  // where a part ends
        #expect(song.clinging(0.3, within: 0.5) == 0 && song.clinging(29.8, within: 0.5) == 30)
    }

    @Test func whatIsInNoPartIsSaidOnlyWhereThereIsSome() {
        #expect(song.hasUnmarked)  // before the first part, and between two
        let whole = Song(
            id: "w", title: "W", artist: nil, lengthS: 20, loudness: [1],
            parts: [
                SongPart(startS: 0, endS: 10, mark: "REST", title: "Rest", colour: .indigo),
                SongPart(startS: 10, endS: 20, mark: "FOCUS", title: "Focus", colour: .orange),
            ],
            cover: nil)
        #expect(!whole.hasUnmarked)
    }

    @Test func howLoudASoundIsBetweenTwoSlicesIsAsFarFromEachAsItLies() {
        let ramp = Song(
            id: "r", title: "R", artist: nil, lengthS: 4, loudness: [0, 1, 0.5, 0.5], parts: [], cover: nil)
        #expect(ramp.loudness(at: 0.125) == 0)  // the middle of the first slice
        #expect(ramp.loudness(at: 0.25) == 0.5)  // halfway to the second
        #expect(ramp.loudness(at: 0.5) == 0.75)
        #expect(ramp.loudness(at: 0) == 0 && ramp.loudness(at: 1) == 0.5)  // the ends: their slices
    }
}
