import Foundation
import XCTest

enum FixtureLoader {
    static func loadJSON<T: Decodable>(
        _ name: String,
        as type: T.Type = T.self,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> T {
        guard let url = Bundle.module.url(forResource: name, withExtension: "json") else {
            XCTFail("Missing fixture \(name).json", file: file, line: line)
            throw FixtureError.missing(name)
        }

        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(T.self, from: data)
    }
}

private enum FixtureError: Error {
    case missing(String)
}
