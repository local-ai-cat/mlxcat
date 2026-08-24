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
        // Library form of the baseline producer, for the on-device iOS
        // benchmark host (pkg 113 ios-row-producer).
        .library(
            name: "MLXCatBaselineKit",
            targets: ["MLXCatBaselineKit"]
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
            // atlas-open-sources/mlx-swift-lm = upstream 01472a78 plus two changes —
            // its mlx-swift dependency resolves to the SAME atlas-open-sources URL the
            // rest of the graph pins (one URL per package identity), and one fix:
            // Gemma4TextAttention rotates with the per-row `cache?.ropeOffset`
            // instead of the scalar `cache?.offset`, so a row with left padding
            // decodes at its own position rather than the padded batch length.
            // Every other family in that library already does this, including
            // the MLXLLM twin of the same attention.
            //
            // Second of two stacked defects; the first was MLX's own RoPE
            // dispatch grid, carried in atlas-open-sources/mlx. `GemmaRoPEOffsetProbeTests`
            // gates both. Return to upstream if it takes the change.
            url: "https://github.com/atlas-open-sources/mlx-swift-lm.git",
            revision: "10ccd66377efbb565d2502e3dc06c3f08b710dfa"  // fusion-exp: b2dfb166 + fused silu MLPs
        ),
        // atlas-open-sources/mlx-swift = 0.31.6 with its vendored mlx submodule
        // moved to our three-line backport of ml-explore/mlx 76a977ca (#3498).
        //
        // Upstream mlx fixed the RoPE single-token batch-grid bug on 2026-05-11:
        // the fast path taken for batched decode with a scalar offset dispatched
        // a grid with no batch axis, so it rotated row 0 and left rows 1..B-1
        // UNROTATED. But mlx-swift 0.31.6 (its newest release) and mlx-swift
        // `main` both vendor mlx at ce45c525 from 2026-03-12, two months before
        // the fix — so no release or branch of mlx-swift reaches it.
        //
        // `RoPEBatchGridProbeTests` is the reproduction and the tripwire.
        // Return this to `.upToNextMinor(from:)` on ml-explore/mlx-swift the
        // moment a release vendors a post-76a977ca mlx.
        .package(
            url: "https://github.com/atlas-open-sources/mlx-swift",
            revision: "934264e56a75799f5784f5900e7bf1c60a3635a9"
        ),
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
        // The measurement core as a library, so an on-device XCTest can produce
        // iOS leaderboard rows through the same arms/corpus/metric as the CLI
        // (pkg 113 ios-row-producer).
        .target(
            name: "MLXCatBaselineKit",
            dependencies: [
                "MLXCat",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MLXCatBaseline",
            dependencies: ["MLXCatBaselineKit"]
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
