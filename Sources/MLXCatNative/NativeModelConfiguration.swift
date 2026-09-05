import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXCat
import MLXCatHTTP
import MLXVLM
import Tokenizers

extension NativeModelLoader {
    static func cacheCapabilities(for configuration: NativeModelKindConfiguration) -> ModelCacheCapabilities {
        ModelCacheCapabilities(
            usesWindowedKVCache: usesWindowedKVCache(configuration: configuration),
            kvCacheProfile: configuration.kvCacheProfile
        )
    }

    private static func usesWindowedKVCache(configuration: NativeModelKindConfiguration) -> Bool {
        let modelType = configuration.modelType.lowercased()
        if windowedKVModelTypes.contains(modelType) {
            return true
        }
        if configuration.useSlidingWindow == false {
            return false
        }
        if let slidingWindow = configuration.slidingWindow, slidingWindow > 0 {
            return true
        }
        if let attentionChunkSize = configuration.attentionChunkSize, attentionChunkSize > 0 {
            return true
        }
        if configuration.layerTypes.contains(where: { $0.localizedCaseInsensitiveContains("sliding") }) {
            return true
        }
        return false
    }

    func processorClass(in modelURL: URL) -> String? {
        for url in processorConfigurationURLs(in: modelURL) {
            guard FileManager.default.fileExists(atPath: url.path) else {
                continue
            }
            guard let data = try? Data(contentsOf: url),
                  let config = try? JSONDecoder.json5().decode(NativeProcessorKindConfiguration.self, from: data)
            else {
                continue
            }
            return config.processorClass
        }
        return nil
    }

    func hasProcessorConfiguration(in modelURL: URL) -> Bool {
        processorConfigurationURLs(in: modelURL).contains { url in
            FileManager.default.fileExists(atPath: url.path)
        }
    }

    private func processorConfigurationURLs(in modelURL: URL) -> [URL] {
        [
            modelURL.appending(component: "preprocessor_config.json"),
            modelURL.appending(component: "processor_config.json")
        ]
    }

}

struct NativeModelKindConfiguration: Decodable {
    let modelType: String
    let slidingWindow: Int?
    let useSlidingWindow: Bool?
    let attentionChunkSize: Int?
    let layerTypes: [String]
    let kvCacheProfile: ModelKVCacheProfile?

    enum CodingKeys: String, Swift.CodingKey {
        case modelType = "model_type"
        case slidingWindow = "sliding_window"
        case useSlidingWindow = "use_sliding_window"
        case attentionChunkSize = "attention_chunk_size"
        case layerTypes = "layer_types"
        case hiddenLayerCount = "num_hidden_layers"
        case attentionHeadCount = "num_attention_heads"
        case keyValueHeadCount = "num_key_value_heads"
        case headDimension = "head_dim"
        case hiddenSize = "hidden_size"
        case torchDType = "torch_dtype"
        case dtype
        case textConfig = "text_config"
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        self.useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow)
        self.attentionChunkSize = try container.decodeIfPresent(Int.self, forKey: .attentionChunkSize)
        self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        let textConfig = try container.decodeIfPresent(NativeTextConfiguration.self, forKey: .textConfig)
        let hiddenLayerCount = try container.decodeIfPresent(Int.self, forKey: .hiddenLayerCount)
            ?? textConfig?.hiddenLayerCount
        let attentionHeadCount = try container.decodeIfPresent(Int.self, forKey: .attentionHeadCount)
            ?? textConfig?.attentionHeadCount
        let keyValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .keyValueHeadCount)
            ?? textConfig?.keyValueHeadCount
        let headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
            ?? textConfig?.headDimension
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? textConfig?.hiddenSize
        let dtype = try container.decodeIfPresent(String.self, forKey: .torchDType)
            ?? container.decodeIfPresent(String.self, forKey: .dtype)
            ?? textConfig?.dtype
        self.kvCacheProfile = ModelKVCacheProfile(
            hiddenLayerCount: hiddenLayerCount,
            attentionHeadCount: attentionHeadCount,
            keyValueHeadCount: keyValueHeadCount,
            headDimension: headDimension,
            hiddenSize: hiddenSize,
            scalarByteCount: ModelKVCacheProfile.scalarByteCount(forDType: dtype)
        )
    }
}

struct NativeTextConfiguration: Decodable {
    let hiddenLayerCount: Int?
    let attentionHeadCount: Int?
    let keyValueHeadCount: Int?
    let headDimension: Int?
    let hiddenSize: Int?
    let dtype: String?

    enum CodingKeys: String, Swift.CodingKey {
        case hiddenLayerCount = "num_hidden_layers"
        case attentionHeadCount = "num_attention_heads"
        case keyValueHeadCount = "num_key_value_heads"
        case headDimension = "head_dim"
        case hiddenSize = "hidden_size"
        case torchDType = "torch_dtype"
        case dtype
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenLayerCount = try container.decodeIfPresent(Int.self, forKey: .hiddenLayerCount)
        self.attentionHeadCount = try container.decodeIfPresent(Int.self, forKey: .attentionHeadCount)
        self.keyValueHeadCount = try container.decodeIfPresent(Int.self, forKey: .keyValueHeadCount)
        self.headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
        self.dtype = try container.decodeIfPresent(String.self, forKey: .torchDType)
            ?? container.decodeIfPresent(String.self, forKey: .dtype)
    }
}

struct NativeProcessorKindConfiguration: Decodable {
    let processorClass: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
    }
}
