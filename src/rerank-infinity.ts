import type { Reranker, RerankResult } from "./types.js";

export class InfinityReranker implements Reranker {
  private readonly model: string;
  private readonly baseUrl: string;

  constructor(model?: string, baseUrl?: string) {
    this.model = model ?? process.env.INFINITY_RERANK_MODEL ?? "mixedbread-ai/mxbai-rerank-xsmall-v1";
    this.baseUrl = (baseUrl ?? process.env.INFINITY_URL ?? "http://localhost:7997").replace(/\/$/, "");
  }

  async rerank(query: string, documents: string[], topK?: number): Promise<RerankResult[]> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/rerank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: this.model,
          query,
          documents,
          top_n: topK ?? documents.length,
        }),
      });
    } catch (err: unknown) {
      if (err instanceof TypeError || (err as any)?.cause?.code === "ECONNREFUSED") {
        throw new Error(`Cannot connect to Infinity at ${this.baseUrl}. Is it running? Start with: bun run infra:up`);
      }
      throw err;
    }

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Infinity API error (${res.status}): ${body}`);
    }

    const data: unknown = await res.json();
    const results = (data as Record<string, unknown>)?.results;
    if (!Array.isArray(results)) {
      throw new Error("Unexpected Infinity response: no results returned");
    }

    return results.map((r: any) => ({
      index: r.index as number,
      relevanceScore: r.relevance_score as number,
    }));
  }
}
