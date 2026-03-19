import type { DenseEmbedder } from "./types.js";

export type CohereModel = "embed-english-v3.0" | "embed-multilingual-v3.0" | "embed-english-light-v3.0";

const MODELS: Record<CohereModel, { dimension: number; maxTokens: number }> = {
  "embed-english-v3.0": { dimension: 1024, maxTokens: 512 },
  "embed-multilingual-v3.0": { dimension: 1024, maxTokens: 512 },
  "embed-english-light-v3.0": { dimension: 384, maxTokens: 512 },
};

export class CohereEmbedder implements DenseEmbedder {
  private client: import("cohere-ai").CohereClient | null = null;
  readonly dimension: number;
  readonly maxTokens: number;
  private readonly model: CohereModel;

  constructor(model: CohereModel = "embed-english-v3.0") {
    this.model = model;
    this.dimension = MODELS[model].dimension;
    this.maxTokens = MODELS[model].maxTokens;
  }

  private async getClient() {
    if (!this.client) {
      if (!process.env.CO_API_KEY) {
        throw new Error("CO_API_KEY environment variable is required for Cohere embeddings");
      }
      const { CohereClient } = await import("cohere-ai");
      this.client = new CohereClient({ token: process.env.CO_API_KEY });
    }
    return this.client;
  }

  // Cohere distinguishes query vs document input types for better retrieval quality
  async embedQuery(text: string): Promise<number[]> {
    const client = await this.getClient();
    const res = await client.v2.embed({
      texts: [text],
      model: this.model,
      inputType: "search_query",
      embeddingTypes: ["float"],
    });
    return res.embeddings.float![0]!;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const client = await this.getClient();
    const results: number[][] = [];

    for (let i = 0; i < texts.length; i += 96) {
      const batch = texts.slice(i, i + 96);
      const res = await client.v2.embed({
        texts: batch,
        model: this.model,
        inputType: "search_document",
        embeddingTypes: ["float"],
      });
      results.push(...res.embeddings.float!);
    }
    return results;
  }
}
