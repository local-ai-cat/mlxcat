import Foundation
import MLXCat
@testable import MLXCatHTTP
@testable import MLXCatHTTPServer
import XCTest

final class RerankTests: XCTestCase {
    func testRerankRequestParsesStringDocumentsWithDefaults() throws {
        let request = try OpenAIRerankRequest.parse(
            Data(
                """
                {
                  "model": "rerank-model",
                  "query": "capital of France",
                  "documents": ["Paris is the capital.", "Bananas are yellow."]
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.model, "rerank-model")
        XCTAssertEqual(request.query, .string("capital of France"))
        XCTAssertEqual(request.documents, [.string("Paris is the capital."), .string("Bananas are yellow.")])
        XCTAssertNil(request.topN)
        XCTAssertTrue(request.returnDocuments)
    }

    func testRerankRequestParsesObjectDocumentsTopNAndReturnDocuments() throws {
        let request = try OpenAIRerankRequest.parse(
            Data(
                """
                {
                  "model": "rerank-model",
                  "query": {"text": "capital of France"},
                  "documents": [{"text": "Paris", "title": "France"}, {"text": "Berlin"}],
                  "top_n": 1,
                  "return_documents": false,
                  "max_chunks_per_doc": 4
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.query, .object(["text": .string("capital of France")]))
        XCTAssertEqual(request.documents[0], .object(["text": .string("Paris"), "title": .string("France")]))
        XCTAssertEqual(request.topN, 1)
        XCTAssertFalse(request.returnDocuments)
        XCTAssertEqual(request.maxChunksPerDoc, 4)
    }

    func testRerankTopNMatchesReferenceSlicingForZeroAndNegativeValues() async throws {
        let backend = FakeRerankBackend(scores: [0.2, 0.9, 0.5])
        let request = OpenAIRerankRequest(
            model: "rerank-model",
            query: .string("capital"),
            documents: [.string("weak"), .string("best"), .string("middle")],
            topN: 0
        )
        let empty = try await backend.rerank(request)
        XCTAssertEqual(empty.indices, [])

        let negative = try await backend.rerank(
            OpenAIRerankRequest(
                model: "rerank-model",
                query: .string("capital"),
                documents: request.documents,
                topN: -1
            )
        )
        XCTAssertEqual(negative.indices, [1, 2])
    }

    func testRerankObjectDocumentsPreserveJSONValues() throws {
        let request = try OpenAIRerankRequest.parse(
            Data(
                """
                {
                  "model": "rerank-model",
                  "query": "capital of France",
                  "documents": [{"text": "Paris", "rank": 1, "fresh": true}]
                }
                """.utf8
            )
        )
        let response = buildRerankResponse(
            request: request,
            result: OpenAIRerankResult(scores: [1], indices: [0], totalTokens: 3)
        )

        let results = try XCTUnwrap(response["results"] as? [[String: Any]])
        let document = try XCTUnwrap(results[0]["document"] as? [String: Any])
        XCTAssertEqual(document["text"] as? String, "Paris")
        XCTAssertEqual(document["rank"] as? Double, 1)
        XCTAssertEqual(document["fresh"] as? Bool, true)
    }

    func testRerankResponseShapeSortAndTopN() async throws {
        let backend = FakeRerankBackend(scores: [0.2, 0.9, 0.5])
        let request = OpenAIRerankRequest(
            model: "rerank-model",
            query: .string("capital"),
            documents: [.string("weak"), .string("best"), .string("middle")],
            topN: 2
        )

        let response = buildRerankResponse(request: request, result: try await backend.rerank(request))

        XCTAssertEqual(response["model"] as? String, "rerank-model")
        XCTAssertTrue((response["id"] as? String)?.hasPrefix("rerank-") == true)
        let results = try XCTUnwrap(response["results"] as? [[String: Any]])
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0]["index"] as? Int, 1)
        XCTAssertEqual(results[0]["relevance_score"] as? Float, 0.9)
        XCTAssertEqual(results[0]["document"] as? [String: String], ["text": "best"])
        XCTAssertEqual(results[1]["index"] as? Int, 2)
        let usage = try XCTUnwrap(response["usage"] as? [String: Int])
        XCTAssertEqual(usage["total_tokens"], 42)
    }

    func testRerankResponseCanReturnNullDocuments() {
        let request = OpenAIRerankRequest(
            model: "rerank-model",
            query: .string("capital"),
            documents: [.string("Paris")],
            returnDocuments: false
        )
        let response = buildRerankResponse(
            request: request,
            result: OpenAIRerankResult(scores: [0.7], indices: [0], totalTokens: 1)
        )

        let results = response["results"] as? [[String: Any]]
        XCTAssertTrue(results?[0]["document"] is NSNull)
    }

    func testRerankUnknownModelReturnsNotFound() throws {
        let response = try XCTUnwrap(
            rerankModelAvailabilityResponse(
                requestedModel: "missing",
                rerankModels: [OpenAIModelInfo(id: "rerank-model")]
            )
        )

        XCTAssertEqual(response.status, 404)
        let error = try XCTUnwrap(response.body["error"] as? [String: Any])
        XCTAssertEqual(error["message"] as? String, "Rerank model 'missing' not found.")
        XCTAssertEqual(error["type"] as? String, "not_found_error")
    }

    func testRerankEmptyDocumentsIsBadRequest() {
        XCTAssertThrowsError(
            try OpenAIRerankRequest.parse(
                Data(
                    """
                    {
                      "model": "rerank-model",
                      "query": "capital",
                      "documents": []
                    }
                    """.utf8
                )
            )
        ) { error in
            XCTAssertEqual(error as? OpenAIHTTPError, OpenAIHTTPError(status: 400, message: "Documents cannot be empty"))
        }
    }

    func testRerankModelDiscoveryClassifiesQwenRerankerDirectories() throws {
        let directory = try temporaryModelDirectory(
            name: "Qwen3-Reranker-0.6B",
            config: [
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
            ]
        )

        XCTAssertTrue(NativeRerankBackend.isRerankModelDirectory(directory))
    }

    func testRerankModelDiscoveryAcceptsExplicitRerankerArchitecturesWithoutNameHint() throws {
        let jina = try temporaryModelDirectory(
            name: "jina-v3",
            config: ["architectures": ["JinaForRanking"]]
        )
        let sequenceClassifier = try temporaryModelDirectory(
            name: "modernbert-classifier",
            config: ["architectures": ["ModernBertForSequenceClassification"]]
        )

        XCTAssertTrue(NativeRerankBackend.isRerankModelDirectory(jina))
        XCTAssertTrue(NativeRerankBackend.isRerankModelDirectory(sequenceClassifier))
    }

    func testNativeRerankCacheStateEvictsPreviousModelBeforeDifferentLoad() throws {
        var state = NativeRerankCacheState()

        XCTAssertNil(try state.prepareLoad(id: "rerank-a", estimatedSize: 100, memoryCeilingBytes: 200))
        state.finishLoad(id: "rerank-a")

        let evicted = try state.prepareLoad(id: "rerank-b", estimatedSize: 100, memoryCeilingBytes: 200)

        XCTAssertEqual(evicted, "rerank-a")
        XCTAssertNil(state.loadedModelID)
        state.finishLoad(id: "rerank-b")
        XCTAssertEqual(state.loadedModelID, "rerank-b")
    }

    func testNativeRerankCacheStateRejectsModelAboveMemoryCeilingWith507() {
        var state = NativeRerankCacheState()

        XCTAssertThrowsError(
            try state.prepareLoad(id: "rerank-huge", estimatedSize: 300, memoryCeilingBytes: 200)
        ) { error in
            XCTAssertEqual(
                error as? OpenAIHTTPError,
                OpenAIHTTPError(
                    status: 507,
                    message: "Model 'rerank-huge' is too large: 300.00B exceeds memory ceiling 200.00B"
                )
            )
        }
    }

    func testNativeRerankSmokeWhenModelGateIsSet() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["MLXSERVE_RERANK_TEST_MODEL"] else {
            throw XCTSkip("Set MLXSERVE_RERANK_TEST_MODEL to run the rerank scoring smoke test.")
        }
        try MLXMetalRuntime.requireAvailable()

        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let model = DiscoveredModel(
            id: modelURL.lastPathComponent,
            modelURL: modelURL,
            estimatedSize: 0
        )
        let backend = NativeRerankBackend(models: [model.id: model])
        let request = OpenAIRerankRequest(
            model: model.id,
            query: .string("capital of France"),
            documents: [.string("Paris is the capital of France."), .string("Bananas are yellow.")],
            topN: 1
        )

        let result = try await backend.rerank(request)

        XCTAssertEqual(result.scores.count, 2)
        XCTAssertEqual(result.indices.count, 1)
        XCTAssertTrue(result.scores.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(result.totalTokens, 0)
    }

    private func temporaryModelDirectory(name: String, config: [String: Any]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))
        return directory
    }
}

private struct FakeRerankBackend: OpenAIRerankBackend {
    let rerankModels = [OpenAIModelInfo(id: "rerank-model")]
    let scores: [Float]

    func rerank(_ request: OpenAIRerankRequest) async throws -> OpenAIRerankResult {
        let sortedIndices = scores.indices.sorted {
            if scores[$0] == scores[$1] {
                return $0 < $1
            }
            return scores[$0] > scores[$1]
        }
        let indices = applyRerankTopN(sortedIndices, topN: request.topN)
        return OpenAIRerankResult(scores: scores, indices: indices, totalTokens: 42)
    }
}
