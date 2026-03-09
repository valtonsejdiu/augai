import { EmbeddingModel, FlagEmbedding } from "fastembed";

// Re-export so consumers can reference the enum without importing fastembed directly
export { EmbeddingModel } from "fastembed";

// Dimension lookup for each supported model.
// Different models produce different vector sizes — you can't mix them in one Qdrant collection.
const MODEL_DIMENSIONS: Record<string, number> = {
  [EmbeddingModel.AllMiniLML6V2]: 384,   // Small, fast, good general purpose
  [EmbeddingModel.BGESmallENV15]: 384,   // Small, newer architecture
  [EmbeddingModel.BGEBaseENV15]: 768,    // Larger, more accurate
  [EmbeddingModel.MLE5Large]: 1024,      // Multilingual, heavy
};

// Why a wrapper? Two reasons:
// 1. FlagEmbedding.init() is async, so you can't use it at import time.
//    This class handles lazy initialization — the model loads once on first use.
// 2. fastembed's embed() returns an AsyncGenerator (for batch streaming).
//    Most of the time we just want number[][] back. This normalizes the API.

export class Embedder {
  private model: FlagEmbedding | null = null;
  readonly modelType: EmbeddingModel;
  readonly dimension: number;

  constructor(modelType: EmbeddingModel = EmbeddingModel.AllMiniLML6V2) {
    this.modelType = modelType;
    this.dimension = MODEL_DIMENSIONS[modelType] ?? 384;
  }

  private async getModel(): Promise<FlagEmbedding> {
    if (!this.model) {
      // Cast needed because FlagEmbedding.init uses a discriminated union
      // that doesn't accept the full EmbeddingModel enum directly
      this.model = await FlagEmbedding.init({
        model: this.modelType as EmbeddingModel.AllMiniLML6V2,
      });
    }
    return this.model;
  }

  // Embed a single query. Use this for search queries.
  async embedQuery(text: string): Promise<number[]> {
    const model = await this.getModel();
    const result = await model.queryEmbed(text);
    // fastembed returns Float32Array, but Qdrant's client expects number[].
    return Array.from(result);
  }

  // Embed multiple texts. Use this for batch inserts.
  async embedBatch(texts: string[]): Promise<number[][]> {
    const model = await this.getModel();
    const results: number[][] = [];
    for await (const batch of model.embed(texts)) {
      results.push(...batch.map((v) => Array.from(v)));
    }
    return results;
  }
}
