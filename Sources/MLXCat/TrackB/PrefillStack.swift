import Foundation
import MLX
import MLXLMCommon

/// A set of cold admissions prefilling together in ONE forward per tick
/// (pkg 115 lever A).
///
/// We prefilled exactly one admission at a time, so the k-th concurrent request
/// waited through k-1 whole prefills before its first token: at c8 that reads
/// as an 8.0-8.3x TTFT spread where mlx-lm's is 1.0-1.05, and our prefill
/// throughput under load sat flat against its own idle rate — no amortization
/// at all (`planning/runs/2026-08-24-concurrency-cliff/REPORT.md`, component d).
/// Stacking k short prompts into one forward is what removes that; round-robin
/// alone would only redistribute the waiting.
///
/// The stack owns the batched per-layer caches. Rows are ragged — they carry
/// different amounts of prompt left — so each tick right-pads to a common width
/// and rolls the padding out again; see
/// ``BatchKVCache/prepare(rightPadding:)``.
final class PrefillStack {
    private var layers: [BatchLayerCache]
    private(set) var rowCount: Int

    private init(layers: [BatchLayerCache], rowCount: Int) {
        self.layers = layers
        self.rowCount = rowCount
    }

    /// Build a stack over `rowCaches`, one entry per row, each a cold cache set
    /// straight from `model.newCache(parameters:)`.
    ///
    /// Returns nil when the model has any layer a stack cannot own — the caller
    /// prefills that model one admission at a time exactly as before. Failing
    /// to a slower correct path is the point: a stack that formed over a layer
    /// whose batch-axis rules it does not implement would corrupt the cache
    /// rather than lose throughput.
    static func form(rowCaches: [[any KVCache]]) -> PrefillStack? {
        guard rowCaches.count > 1, let firstRow = rowCaches.first else { return nil }
        let layerCount = firstRow.count
        guard layerCount > 0, rowCaches.allSatisfy({ $0.count == layerCount }) else { return nil }

        var layers: [BatchLayerCache] = []
        layers.reserveCapacity(layerCount)
        for layerIndex in 0 ..< layerCount {
            // Every row's layer must be cold, not just the exemplar: a warm row
            // carries tokens that emptyBatched would silently drop on the floor.
            guard rowCaches.allSatisfy({ $0[layerIndex].state.isEmpty }) else { return nil }
            guard
                let layer = BatchLayerCache.emptyBatched(
                    rows: rowCaches.count,
                    like: firstRow[layerIndex]
                )
            else { return nil }
            layers.append(layer)
        }
        return PrefillStack(layers: layers, rowCount: rowCaches.count)
    }

    /// The cache set to hand the model for the batched forward.
    var caches: [any KVCache] {
        layers.map(\.kvCache)
    }

    /// Declare row `i`'s write `padding[i]` positions short of the batch width.
    func prepare(rightPadding padding: [Int]) throws {
        for layer in layers {
            try layer.prepare(rightPadding: padding)
        }
    }

    /// Roll each row's padding out of the tip column.
    func finalize() {
        for layer in layers {
            layer.finalize()
        }
    }

    /// Row `i`'s own cache set, for handing a finished prefill to decode.
    func extractRow(_ row: Int) -> [any KVCache] {
        layers.map { $0.extract(row) }
    }

    /// Drop rows that have left the stack. Indices are into the CURRENT row
    /// order; the caller re-indexes its own bookkeeping to match.
    func removeRows(_ rows: [Int]) {
        let dropped = Set(rows)
        guard !dropped.isEmpty else { return }
        let kept = (0 ..< rowCount).filter { !dropped.contains($0) }
        for layer in layers {
            layer.filter(keeping: kept)
        }
        rowCount = kept.count
    }

    /// Right-pad ragged per-row chunks into one `[rows, width]` token block.
    ///
    /// Padding is zeros and is never read: it sits AFTER a row's real tokens,
    /// causal attention cannot look forward, and ``finalize()`` rolls it out of
    /// the cache before decode ever sees the row.
    static func tokenBlock(chunks: [[Int]]) -> (block: MLXArray, padding: [Int])? {
        guard !chunks.isEmpty else { return nil }
        let lengths = chunks.map(\.count)
        guard let width = lengths.max(), width > 0 else { return nil }

        var flattened: [Int32] = []
        flattened.reserveCapacity(chunks.count * width)
        for chunk in chunks {
            flattened.append(contentsOf: chunk.map(Int32.init))
            flattened.append(contentsOf: Array(repeating: Int32(0), count: width - chunk.count))
        }
        return (MLXArray(flattened, [chunks.count, width]), lengths.map { width - $0 })
    }
}
