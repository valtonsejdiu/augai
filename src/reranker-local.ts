import type { Reranker, RerankResult } from "./types.js";

const K1 = 1.5;
const B = 0.75;

function tokenize(text: string): string[] {
  return text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

export class LocalReranker implements Reranker {
  async rerank(query: string, documents: string[], topK?: number): Promise<RerankResult[]> {
    if (documents.length === 0) return [];

    const queryTerms = tokenize(query);
    const tokenized = documents.map(tokenize);
    const avgDocLen = tokenized.reduce((s, d) => s + d.length, 0) / tokenized.length;

    const tfMaps = tokenized.map((tokens) => {
      const tf = new Map<string, number>();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      return tf;
    });

    const idf = new Map<string, number>();
    for (const term of queryTerms) {
      if (idf.has(term)) continue;
      let df = 0;
      for (const tf of tfMaps) if (tf.has(term)) df++;
      idf.set(term, Math.log((documents.length - df + 0.5) / (df + 0.5) + 1));
    }

    const rawScores = tokenized.map((tokens, i) => {
      const tf = tfMaps[i]!;
      const docLen = tokens.length;
      let score = 0;
      for (const term of queryTerms) {
        const freq = tf.get(term) ?? 0;
        if (freq === 0) continue;
        const termIdf = idf.get(term)!;
        const numerator = freq * (K1 + 1);
        const denominator = freq + K1 * (1 - B + B * (docLen / avgDocLen));
        score += termIdf * (numerator / denominator);
      }
      return score;
    });

    const max = rawScores.reduce((a, b) => Math.max(a, b), 0);
    const results: RerankResult[] = rawScores.map((score, index) => ({
      index,
      relevanceScore: max === 0 ? 0 : score / max,
    }));

    results.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return topK !== undefined ? results.slice(0, topK) : results;
  }
}
