import Foundation

struct InklingConfiguration: Decodable {
    struct Text: Decodable {
        struct Loop: Decodable {
            let start: Int
            let end: Int
            let passes: Int
            let gateRank: Int
            let isVariable: Bool

            enum CodingKeys: String, CodingKey {
                case start = "core_start"
                case end = "core_end"
                case passes = "num_passes"
                case gateRank = "gate_rank"
                case isVariable = "is_varloop"
            }

            init(from decoder: any Decoder) throws {
                let values = try decoder.container(keyedBy: CodingKeys.self)
                start = try values.decode(Int.self, forKey: .start)
                end = try values.decode(Int.self, forKey: .end)
                passes = try values.decode(Int.self, forKey: .passes)
                gateRank = try values.decode(Int.self, forKey: .gateRank)
                isVariable = try values.decodeIfPresent(Bool.self, forKey: .isVariable) ?? false
            }
        }

        let rmsEpsilon: Float
        let heads: Int
        let keyValueHeads: Int
        let headWidth: Int
        let relativeWidth: Int
        let queryClip: Float
        let relativeClip: Float
        let logScaleFloor: Int?
        let logScaleAlpha: Float
        let normalizeResidual: Bool
        let routeCount: Int
        let routeScale: Float
        let upCap: Float
        let gateCap: Float
        let logitCap: Float
        let vocabularySize: Int
        let hiddenSize: Int
        let layerCount: Int
        let residualBlockSize: Int
        let localLayerIDs: [Int]
        let windowSize: Int
        let convolutionSize: Int
        let loop: Loop

        enum CodingKeys: String, CodingKey {
            case rmsEpsilon = "rms_norm_eps"
            case heads = "num_attention_heads"
            case keyValueHeads = "num_key_value_heads"
            case headWidth = "head_dim"
            case relativeWidth = "d_rel"
            case queryClip = "qk_l2_clip_threshold"
            case relativeClip = "rel_l2_clip_threshold"
            case logScaleFloor = "log_scaling_n_floor"
            case logScaleAlpha = "log_scaling_alpha"
            case normalizeResidual = "attn_res_normalize_values"
            case routeCount = "num_experts_per_tok"
            case routeScale = "route_scale"
            case upCap = "mlp_up_softcap"
            case gateCap = "mlp_gate_softcap"
            case logitCap = "final_logit_softcapping"
            case vocabularySize = "unpadded_vocab_size"
            case hiddenSize = "hidden_size"
            case layerCount = "num_hidden_layers"
            case residualBlockSize = "n_layers_per_attn_res_block"
            case localLayerIDs = "local_layer_ids"
            case windowSize = "sliding_window_size"
            case convolutionSize = "sconv_kernel_size"
            case loop = "loop_config"
        }

        func makeLoop() throws -> InklingLoop {
            guard loop.start < loop.end, loop.gateRank > 0, !loop.isVariable else {
                throw InklingLoop.ConfigurationError.invalidLoop
            }
            return try InklingLoop(
                layerCount: layerCount, core: loop.start..<loop.end,
                passes: loop.passes, residualBlockSize: residualBlockSize)
        }
    }

    struct Vision: Decodable {
        let patchSize: Int
        let merge: Int
        let width: Int
        let layers: Int
        let heads: Int
        let gridSize: Int
        let epsilon: Float

        enum CodingKeys: String, CodingKey {
            case patchSize = "native_patch_size"
            case merge = "adapter_merge_size"
            case width = "hidden_size"
            case layers = "num_hidden_layers"
            case heads = "num_attention_heads"
            case gridSize = "posemb_grid_size"
            case epsilon = "layer_norm_eps"
        }
    }

    let text: Text
    let vision: Vision?
    let eosTokenID: Int?
    let imageTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case eosTokenID = "eos_token_id"
        case imageTokenID = "image_token_id"
    }
}
