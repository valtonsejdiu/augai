import type { DenseEmbedder, SparseEmbedder, SparseVector, Reranker, RerankResult } from "./types.js";

const BASE_URL = (process.env.AUGAI_EMBED_URL ?? "http://localhost:8081").replace(/\/$/, "");

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Sidecar ${path} failed: ${res.status} ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export class SidecarDenseEmbedder implements DenseEmbedder {
  readonly dimension = 768;
  readonly maxTokens = 8192;

  async embedQuery(text: string): Promise<number[]> {
    const res = await post<{ vectors: number[][] }>("/embed/dense", { texts: [text] });
    return res.vectors[0]!;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const res = await post<{ vectors: number[][] }>("/embed/dense", { texts });
    return res.vectors;
  }
}

export class SidecarSparseEmbedder implements SparseEmbedder {
  async embedQuery(text: string): Promise<SparseVector> {
    const res = await post<{ vectors: SparseVector[] }>("/embed/sparse", { texts: [text] });
    return res.vectors[0]!;
  }

  async embedBatch(texts: string[]): Promise<SparseVector[]> {
    const res = await post<{ vectors: SparseVector[] }>("/embed/sparse", { texts });
    return res.vectors;
  }
}

export class SidecarReranker implements Reranker {
  async rerank(query: string, documents: string[], topK?: number): Promise<RerankResult[]> {
    const res = await post<{ results: { index: number; score: number }[] }>("/rerank", {
      query,
      passages: documents,
      ...(topK !== undefined && { top_k: topK }),
    });
    return res.results.map((r) => ({ index: r.index, relevanceScore: r.score }));
  }
}

export async function checkSidecarHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch {
    return false;
  }
}
