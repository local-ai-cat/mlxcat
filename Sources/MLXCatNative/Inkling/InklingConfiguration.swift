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

        let hiddenSize: Int
        let layerCount: Int
        let residualBlockSize: Int
        let localLayerIDs: [Int]
        let windowSize: Int
        let convolutionSize: Int
        let loop: Loop

        enum CodingKeys: String, CodingKey {
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

    let text: Text

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
    }
}
