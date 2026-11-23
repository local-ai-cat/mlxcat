import CoreImage
import MLX
import XCTest

@testable import MLXCatNative

final class InklingVisionTests: XCTestCase {
    func testImagePatchPreservesThePixelAndPadsWithTheReferenceSentinel() throws {
        try MLXMetalRuntime.requireAvailable()
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
        let (tiles, grid) = try InklingImageProcessor.prepare(.ciImage(image), patch: 2)
        XCTAssertEqual(grid.h, 1)
        XCTAssertEqual(grid.w, 1)
        XCTAssertEqual(tiles.shape, [1, 2, 2, 3])
        XCTAssertEqual(tiles[0, 0, 0, 0...].asArray(Float.self), [1, -1, -1])
        XCTAssertEqual(tiles[0, 1, 1, 1].item(Float.self), -1 - 2 / 255, accuracy: 1e-6)
    }
}

extension InklingVisionTests {
    func testImageRowsKeepTheBitmapOrder() throws {
        try MLXMetalRuntime.requireAvailable()
        let bytes = Data([255, 0, 0, 255, 0, 0, 255, 255])
        let image = CIImage(
            bitmapData: bytes, bytesPerRow: 4, size: CGSize(width: 1, height: 2),
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        let (tiles, _) = try InklingImageProcessor.prepare(.ciImage(image), patch: 2)
        XCTAssertEqual(tiles[0, 0, 0, 0...].asArray(Float.self), [1, -1, -1])
        XCTAssertEqual(tiles[0, 1, 0, 0...].asArray(Float.self), [-1, -1, 1])
    }
}
