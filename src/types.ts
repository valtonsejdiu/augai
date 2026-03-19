export interface ChunkOptions {
  maxChunkSize?: number;
  overlapSize?: number;
}

export interface Chunk {
  text: string;
  index: number;
  totalChunks: number;
}

export interface DenseEmbedder {
  readonly dimension: number;
  readonly maxTokens: number;
  embedQuery(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

export interface SparseVector {
  indices: number[];
  values: number[];
}

export interface SparseEmbedder {
  embedQuery(text: string): Promise<SparseVector>;
  embedBatch(texts: string[]): Promise<SparseVector[]>;
}

export interface RerankResult {
  index: number;
  relevanceScore: number;
}

export interface Reranker {
  rerank(query: string, documents: string[], topK?: number): Promise<RerankResult[]>;
}
