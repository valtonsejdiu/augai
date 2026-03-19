import { EmbeddingModel, FlagEmbedding } from "fastembed";
import type { DenseEmbedder } from "./types.js";

export { EmbeddingModel } from "fastembed";

const MODEL_DIMENSIONS: Record<string, number> = {
  [EmbeddingModel.AllMiniLML6V2]: 384,
  [EmbeddingModel.BGESmallENV15]: 384,
  [EmbeddingModel.BGEBaseENV15]: 768,
  [EmbeddingModel.MLE5Large]: 1024,
};

const MODEL_MAX_TOKENS: Record<string, number> = {
  [EmbeddingModel.AllMiniLML6V2]: 256,
  [EmbeddingModel.BGESmallENV15]: 512,
  [EmbeddingModel.BGEBaseENV15]: 512,
  [EmbeddingModel.MLE5Large]: 512,
};

export class FastEmbedEmbedder implements DenseEmbedder {
  private model: FlagEmbedding | null = null;
  readonly modelType: EmbeddingModel;
  readonly dimension: number;
  readonly maxTokens: number;

  constructor(modelType: EmbeddingModel = EmbeddingModel.AllMiniLML6V2) {
    this.modelType = modelType;
    this.dimension = MODEL_DIMENSIONS[modelType] ?? 384;
    this.maxTokens = MODEL_MAX_TOKENS[modelType] ?? 512;
  }

  private async getModel(): Promise<FlagEmbedding> {
    if (!this.model) {
      // Cast needed: FlagEmbedding.init uses a discriminated union that doesn't accept
      // the full EmbeddingModel enum directly
      this.model = await FlagEmbedding.init({
        model: this.modelType as EmbeddingModel.AllMiniLML6V2,
      });
    }
    return this.model;
  }

  async embedQuery(text: string): Promise<number[]> {
    const model = await this.getModel();
    const result = await model.queryEmbed(text);
    return Array.from(result);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const model = await this.getModel();
    const results: number[][] = [];
    for await (const batch of model.embed(texts)) {
      results.push(...batch.map((v) => Array.from(v)));
    }
    return results;
  }
}
