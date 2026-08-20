@testable import MLXCatHTTP
import XCTest

final class HealthTests: XCTestCase {
    func testHealthResponseShapeIncludesDefaultModelAndEnginePoolCounts() throws {
        let response = buildHealthResponse(
            OpenAIHealthInfo(
                defaultModel: "test-model",
                enginePool: OpenAIHealthEnginePool(modelCount: 2, loadedCount: 1)
            )
        )

        XCTAssertEqual(response["status"] as? String, "healthy")
        XCTAssertEqual(response["default_model"] as? String, "test-model")
        let enginePool = try XCTUnwrap(response["engine_pool"] as? [String: Int])
        XCTAssertEqual(enginePool["model_count"], 2)
        XCTAssertEqual(enginePool["loaded_count"], 1)
    }

    func testHealthResponseUsesNullsWhenDetailsAreUnavailable() {
        let response = buildHealthResponse(
            OpenAIHealthInfo(defaultModel: nil, enginePool: nil)
        )

        XCTAssertEqual(response["status"] as? String, "healthy")
        XCTAssertTrue(response["default_model"] is NSNull)
        XCTAssertTrue(response["engine_pool"] is NSNull)
    }
}
