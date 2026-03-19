import { QdrantClient } from "@qdrant/js-client-rest";
import type { ChunkOptions, DenseEmbedder, Reranker, SparseEmbedder } from "./types.js";
import { chunkText } from "./chunker.js";

export interface MemoryPayload {
  text: string;
  sessionId: string;
  role: string;
  timestamp: number;
  sourceId?: string;
  chunkIndex?: number;
  totalChunks?: number;
}

export interface MemoryResult {
  text: string;
  role: string;
  score: number;
  cosineScore: number;
  recencyScore: number;
  rerankScore?: number;
  sparseScore?: number;
  timestamp: number;
}

export interface SearchOptions {
  limit?: number;
  sessionId?: string;
  timeWeight?: boolean;
  alpha?: number;
  decayRate?: number;
  rerank?: boolean;
}

const MS_PER_HOUR = 3_600_000;

export class SessionMemory {
  private client: QdrantClient;
  private dense: DenseEmbedder;
  private sparse: SparseEmbedder | null;
  private reranker: Reranker | null;
  private collectionName: string;
  private initialized = false;
  private readonly chunkDefaults: ChunkOptions;

  constructor(
    client: QdrantClient,
    dense: DenseEmbedder,
    collectionName = process.env.AUGAI_COLLECTION ?? "session_memory",
    sparse: SparseEmbedder | null = null,
    reranker: Reranker | null = null
  ) {
    this.client = client;
    this.dense = dense;
    this.sparse = sparse;
    this.reranker = reranker;
    this.collectionName = collectionName;
    const maxChunkSize = dense.maxTokens * 4;
    this.chunkDefaults = {
      maxChunkSize,
      overlapSize: Math.max(200, Math.floor(maxChunkSize * 0.1)),
    };
  }

  private async ensureCollection(): Promise<void> {
    if (this.initialized) return;

    const collections = await this.client.getCollections();
    const exists = collections.collections.some((c) => c.name === this.collectionName);

    if (!exists) {
      await this.client.createCollection(this.collectionName, {
        vectors: { dense: { size: this.dense.dimension, distance: "Cosine" } },
        ...(this.sparse && {
          sparse_vectors: { sparse: { index: { type: "plain" } } },
        }),
      });
      console.log(`[memory] Created collection: ${this.collectionName}`);
    }

    this.initialized = true;
  }

  async add(
    text: string,
    sessionId: string,
    role: string,
    chunkOptions?: ChunkOptions
  ): Promise<void> {
    await this.ensureCollection();

    const chunks = chunkText(text, chunkOptions ?? this.chunkDefaults);
    const now = Date.now();
    const sourceId = chunks.length > 1 ? crypto.randomUUID() : undefined;
    const texts = chunks.map((c) => c.text);
    const denseVecs = await this.dense.embedBatch(texts);
    const sparseVecs = this.sparse ? await this.sparse.embedBatch(texts) : null;

    await this.client.upsert(this.collectionName, {
      points: chunks.map((chunk, i) => ({
        id: crypto.randomUUID(),
        vector: {
          dense: denseVecs[i]!,
          ...(sparseVecs && { sparse: sparseVecs[i]! }),
        },
        payload: {
          text: chunk.text,
          sessionId,
          role,
          timestamp: now,
          ...(sourceId && {
            sourceId,
            chunkIndex: chunk.index,
            totalChunks: chunk.totalChunks,
          }),
        } satisfies MemoryPayload,
      })),
    });
  }

  async addBatch(
    entries: { text: string; role: string }[],
    sessionId: string,
    chunkOptions?: ChunkOptions
  ): Promise<void> {
    await this.ensureCollection();

    const now = Date.now();
    const allPoints: Omit<MemoryPayload, "sessionId">[] = [];

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i]!;
      const chunks = chunkText(entry.text, chunkOptions ?? this.chunkDefaults);

      if (chunks.length === 1) {
        allPoints.push({ text: chunks[0]!.text, role: entry.role, timestamp: now + i });
      } else {
        const sourceId = crypto.randomUUID();
        for (const chunk of chunks) {
          allPoints.push({
            text: chunk.text,
            role: entry.role,
            timestamp: now + i,
            sourceId,
            chunkIndex: chunk.index,
            totalChunks: chunk.totalChunks,
          });
        }
      }
    }

    const texts = allPoints.map((p) => p.text);
    const denseVecs = await this.dense.embedBatch(texts);
    const sparseVecs = this.sparse ? await this.sparse.embedBatch(texts) : null;

    await this.client.upsert(this.collectionName, {
      points: allPoints.map((point, i) => ({
        id: crypto.randomUUID(),
        vector: {
          dense: denseVecs[i]!,
          ...(sparseVecs && { sparse: sparseVecs[i]! }),
        },
        payload: { ...point, sessionId } satisfies MemoryPayload,
      })),
    });
  }

  private async denseSearch(
    vector: number[],
    fetchLimit: number,
    filter?: object
  ) {
    return this.client.search(this.collectionName, {
      vector: { name: "dense", vector },
      limit: fetchLimit,
      filter,
      with_payload: true,
    });
  }

  private async hybridSearch(
    denseVec: number[],
    sparseVec: { indices: number[]; values: number[] },
    fetchLimit: number,
    filter?: object
  ) {
    const result = await this.client.query(this.collectionName, {
      prefetch: [
        { query: denseVec, using: "dense", limit: fetchLimit },
        { query: sparseVec, using: "sparse", limit: fetchLimit },
      ],
      query: { fusion: "rrf" },
      limit: fetchLimit,
      filter,
      with_payload: true,
      with_vector: false,
    });
    return result.points;
  }

  async search(query: string, options: SearchOptions = {}): Promise<MemoryResult[]> {
    await this.ensureCollection();

    const {
      limit = 5,
      sessionId,
      timeWeight = true,
      alpha = 0.7,
      decayRate = 24,
      rerank = !!this.reranker,
    } = options;

    const denseVec = await this.dense.embedQuery(query);
    const sparseVec = this.sparse ? await this.sparse.embedQuery(query) : null;

    const filter = sessionId
      ? { must: [{ key: "sessionId", match: { value: sessionId } }] }
      : undefined;

    const useReranker = rerank && this.reranker;
    const fetchLimit = Math.min(
      useReranker ? limit * 10 : timeWeight ? limit * 3 : limit,
      1000
    );

    const raw = sparseVec
      ? await this.hybridSearch(denseVec, sparseVec, fetchLimit, filter)
      : await this.denseSearch(denseVec, fetchLimit, filter);

    let rerankScores: Map<number, number> | undefined;
    if (useReranker && raw.length > 0) {
      const texts = raw.map((r) => (r.payload as unknown as MemoryPayload).text);
      const rerankResults = await this.reranker!.rerank(query, texts);
      rerankScores = new Map(rerankResults.map((r) => [r.index, r.relevanceScore]));
    }

    const now = Date.now();

    const scored: MemoryResult[] = raw.map((r, i) => {
      const p = r.payload as unknown as MemoryPayload;
      const cosineScore = r.score;
      const recencyScore = Math.exp(-(now - p.timestamp) / (decayRate * MS_PER_HOUR));

      const rawRerankScore = rerankScores?.get(i);
      // Sigmoid-normalize rerank scores before blending: maps any real → (0,1)
      const rerankScore =
        rawRerankScore !== undefined ? 1 / (1 + Math.exp(-rawRerankScore)) : undefined;

      const relevance = rerankScore ?? cosineScore;
      const score = timeWeight ? relevance * alpha + recencyScore * (1 - alpha) : relevance;

      return {
        text: p.text,
        role: p.role,
        score,
        cosineScore,
        recencyScore,
        ...(rerankScore !== undefined && { rerankScore }),
        timestamp: p.timestamp,
      };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  async listSession(sessionId: string, limit = 20): Promise<MemoryResult[]> {
    await this.ensureCollection();

    const { points } = await this.client.scroll(this.collectionName, {
      filter: { must: [{ key: "sessionId", match: { value: sessionId } }] },
      limit,
      with_payload: true,
      with_vector: false,
    });

    return points
      .map((p) => {
        const payload = p.payload as unknown as MemoryPayload;
        return {
          text: payload.text,
          role: payload.role,
          score: 0,
          cosineScore: 0,
          recencyScore: 0,
          timestamp: payload.timestamp,
        };
      })
      .sort((a, b) => a.timestamp - b.timestamp);
  }

  async clearSession(sessionId: string): Promise<void> {
    await this.ensureCollection();

    await this.client.delete(this.collectionName, {
      filter: {
        must: [{ key: "sessionId", match: { value: sessionId } }],
      },
    });
  }
}
