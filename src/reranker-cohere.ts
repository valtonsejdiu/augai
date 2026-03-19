import type { Reranker, RerankResult } from "./types.js";

export type CohereRerankModel = "rerank-v3.5" | "rerank-multilingual-v3.5";

export class CohereReranker implements Reranker {
  private client: any = null;
  private readonly model: CohereRerankModel;

  constructor(model: CohereRerankModel = "rerank-v3.5") {
    this.model = model;
  }

  private async getClient() {
    if (!this.client) {
      if (!process.env.CO_API_KEY) {
        throw new Error("CO_API_KEY environment variable is required for Cohere reranking");
      }
      const { CohereClient } = await import("cohere-ai");
      this.client = new CohereClient({ token: process.env.CO_API_KEY });
    }
    return this.client;
  }

  async rerank(query: string, documents: string[], topK?: number): Promise<RerankResult[]> {
    if (documents.length === 0) return [];
    const client = await this.getClient();
    const res = await client.v2.rerank({
      query,
      documents,
      model: this.model,
      ...(topK !== undefined && { topN: topK }),
    });
    return res.results.map((item: { index: number; relevanceScore: number }) => ({
      index: item.index,
      relevanceScore: item.relevanceScore,
    }));
  }
}
