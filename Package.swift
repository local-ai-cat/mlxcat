// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "mlxserve",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "MLXServe",
            targets: ["MLXServe"]
        ),
        .library(
            name: "MLXServeHTTP",
            targets: ["MLXServeHTTP"]
        ),
        // The native MLX chat engine wired as an OpenAI-compatible backend,
        // linkable in-process (e.g. by the iOS app) without the server executable
        // or WhisperKit. This is what makes "embed the real engine" possible.
        .library(
            name: "MLXServeNative",
            targets: ["MLXServeNative"]
        ),
        .library(
            name: "MLXServeSpeech",
            targets: ["MLXServeSpeech"]
        ),
        .library(
            name: "MLXServeSpeechWhisperKit",
            targets: ["MLXServeSpeechWhisperKit"]
        ),
        .executable(
            name: "mlxserve-bench",
            targets: ["MLXServeBench"]
        ),
        .executable(
            name: "mlxserve-http",
            targets: ["MLXServeHTTPServer"]
        )
    ],
    dependencies: [
        // Pinned by revision, not tag: 3.31.4 is the newest upstream TAG but it
        // predates the Gemma 4 loader fixes (#384/#390 build the KV-shared tail
        // without k_proj/v_proj; #408 loads shards by index). Verified 2026-07-21
        // that gemma-4-12B/E2B/E4B QAT all load and generate at this revision and
        // fail to load before it. Repin to a tag once upstream cuts one that
        // contains fd0f13b.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            revision: "01472a78fca830689ff78246a82c6d31ab111a78"
        ),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        // Same URL+version as the Local AI Chat app pins — one URL per package
        // identity across the combined graph (SwiftPM escalates the mismatch to
        // an error in future versions). Moved off the atlas-open-sources fork to
        // upstream 3.31.4 (2026-07-19): upstream now ships gemma4_unified natively.
        .package(url: "https://github.com/atlas-open-sources/swift-transformers", revision: "089cb3f02a1718b2943c7e7c4553876cd51a75d1"),
        // Pinned to main by revision: tagged releases (≤0.17) pin swift-transformers
        // <1.2 which conflicts with mlx-swift-lm's ≥1.3; main dropped the dep.
        .package(
            url: "https://github.com/argmaxinc/WhisperKit",
            revision: "dcf3a00f0ae4d5b57bc0aad92063b102b70d5fd1"
        ),
    ],
    targets: [
        .target(
            name: "MLXServe",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "MLXServeBench",
            dependencies: [
                "MLXServe",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXServeHTTP",
            dependencies: []
        ),
        .target(
            name: "MLXServeNative",
            dependencies: [
                "MLXServe",
                "MLXServeHTTP",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXServeSpeech",
            dependencies: []
        ),
        .target(
            name: "MLXServeSpeechWhisperKit",
            dependencies: [
                "MLXServeSpeech",
                .product(name: "WhisperKit", package: "WhisperKit"),
            ]
        ),
        .executableTarget(
            name: "MLXServeHTTPServer",
            dependencies: [
                "MLXServe",
                "MLXServeHTTP",
                "MLXServeNative",
                "MLXServeSpeech",
                "MLXServeSpeechWhisperKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "MLXServeTests",
            dependencies: [
                "MLXServe",
                "MLXServeHTTP",
                "MLXServeNative",
                "MLXServeHTTPServer",
                "MLXServeSpeech",
                "MLXServeSpeechWhisperKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            resources: [
                .process("Fixtures")
            ]
        ),
    ]
)
