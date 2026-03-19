import type { Reranker, RerankResult } from "./types.js";

export type VoyageRerankModel = "rerank-2" | "rerank-2-lite" | "rerank-1" | "rerank-lite-1";

export class VoyageReranker implements Reranker {
  private client: any = null;
  private readonly model: VoyageRerankModel;

  constructor(model: VoyageRerankModel = "rerank-2") {
    this.model = model;
  }

  private async getClient() {
    if (!this.client) {
      if (!process.env.VOYAGEAI_API_KEY) {
        throw new Error("VOYAGEAI_API_KEY environment variable is required for Voyage reranking");
      }
      const { VoyageAIClient } = await import("voyageai");
      this.client = new VoyageAIClient({ apiKey: process.env.VOYAGEAI_API_KEY });
    }
    return this.client;
  }

  async rerank(query: string, documents: string[], topK?: number): Promise<RerankResult[]> {
    if (documents.length === 0) return [];
    const client = await this.getClient();
    const res = await client.rerank({
      query,
      documents,
      model: this.model,
      ...(topK !== undefined && { topK }),
    });
    return (res.data ?? []).map((item: { index: number; relevance_score?: number }) => ({
      index: item.index,
      relevanceScore: item.relevance_score ?? 0,
    }));
  }
}
