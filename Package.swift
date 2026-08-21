// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "mlxcat",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "MLXCat",
            targets: ["MLXCat"]
        ),
        .library(
            name: "MLXCatHTTP",
            targets: ["MLXCatHTTP"]
        ),
        // The native MLX chat engine wired as an OpenAI-compatible backend,
        // linkable in-process (e.g. by the iOS app) without the server executable
        // or WhisperKit. This is what makes "embed the real engine" possible.
        .library(
            name: "MLXCatNative",
            targets: ["MLXCatNative"]
        ),
        .library(
            name: "MLXCatSpeech",
            targets: ["MLXCatSpeech"]
        ),
        .library(
            name: "MLXCatSpeechWhisperKit",
            targets: ["MLXCatSpeechWhisperKit"]
        ),
        .executable(
            name: "mlxcat-bench",
            targets: ["MLXCatBench"]
        ),
        .executable(
            name: "mlxcat-baseline",
            targets: ["MLXCatBaseline"]
        ),
        .executable(
            name: "mlxcat-http",
            targets: ["MLXCatHTTPServer"]
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
            name: "MLXCat",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "MLXCatBench",
            dependencies: [
                "MLXCat",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MLXCatBaseline",
            dependencies: [
                "MLXCat",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXCatHTTP",
            dependencies: []
        ),
        .target(
            name: "MLXCatNative",
            dependencies: [
                "MLXCat",
                "MLXCatHTTP",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "MLXCatSpeech",
            dependencies: []
        ),
        .target(
            name: "MLXCatSpeechWhisperKit",
            dependencies: [
                "MLXCatSpeech",
                .product(name: "WhisperKit", package: "WhisperKit"),
            ]
        ),
        .executableTarget(
            name: "MLXCatHTTPServer",
            dependencies: [
                "MLXCat",
                "MLXCatHTTP",
                "MLXCatNative",
                "MLXCatSpeech",
                "MLXCatSpeechWhisperKit",
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
            name: "MLXCatTests",
            dependencies: [
                "MLXCat",
                "MLXCatHTTP",
                "MLXCatNative",
                "MLXCatHTTPServer",
                "MLXCatSpeech",
                "MLXCatSpeechWhisperKit",
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
