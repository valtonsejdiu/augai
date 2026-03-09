import { QdrantClient } from "@qdrant/js-client-rest";
import { Embedder } from "./embeddings.js";
import { chunkText, type ChunkOptions } from "./chunker.js";

// Each memory entry stored in Qdrant carries this metadata.
export interface MemoryPayload {
  text: string;        // The original text
  sessionId: string;   // Which session this belongs to
  role: string;        // "user" | "assistant" | "system" — who said it
  timestamp: number;   // Unix ms — enables time-based retrieval
  sourceId?: string;   // Links chunks from the same input
  chunkIndex?: number; // Position within source (0-based)
  totalChunks?: number; // How many chunks total
}

// What you get back from a search.
export interface MemoryResult {
  text: string;
  role: string;
  score: number;         // Final blended score (or pure cosine if timeWeight=false)
  cosineScore: number;   // Raw cosine similarity from Qdrant
  recencyScore: number;  // Exponential decay based on age (1.0 = just now, ~0.37 at decayRate hours)
  timestamp: number;
}

export interface SearchOptions {
  limit?: number;
  sessionId?: string;
  timeWeight?: boolean;  // Enable time-weighted scoring (default: true)
  alpha?: number;        // Balance: 0=pure recency, 1=pure cosine (default: 0.7)
  decayRate?: number;    // Hours until recency drops to ~37% (default: 24)
}

export class SessionMemory {
  private client: QdrantClient;
  private embedder: Embedder;
  private collectionName: string;
  private initialized = false;

  // Why pass embedder in? Dependency injection.
  // If you later swap to OpenAI embeddings, you only change the caller,
  // not this class. This is the "D" in SOLID.
  constructor(
    client: QdrantClient,
    embedder: Embedder,
    collectionName = "session_memory"
  ) {
    this.client = client;
    this.embedder = embedder;
    this.collectionName = collectionName;
  }

  // Ensures the collection exists. Idempotent — safe to call multiple times.
  private async ensureCollection(): Promise<void> {
    if (this.initialized) return;

    const collections = await this.client.getCollections();
    const exists = collections.collections.some(
      (c) => c.name === this.collectionName
    );

    if (!exists) {
      await this.client.createCollection(this.collectionName, {
        vectors: {
          size: this.embedder.dimension,
          distance: "Cosine",
        },
      });
      console.log(`[memory] Created collection: ${this.collectionName}`);
    }

    this.initialized = true;
  }

  // Store a message in memory. Long text is automatically chunked into
  // overlapping pieces, each embedded and stored separately with a shared
  // sourceId so they can be reassembled later.
  async add(
    text: string,
    sessionId: string,
    role: string,
    chunkOptions?: ChunkOptions
  ): Promise<void> {
    await this.ensureCollection();

    const chunks = chunkText(text, chunkOptions);
    const now = Date.now();

    if (chunks.length === 1) {
      // Short text — store directly, no chunk metadata needed
      const vector = await this.embedder.embedQuery(text);
      await this.client.upsert(this.collectionName, {
        points: [
          {
            id: crypto.randomUUID(),
            vector,
            payload: {
              text,
              sessionId,
              role,
              timestamp: now,
            } satisfies MemoryPayload,
          },
        ],
      });
      return;
    }

    // Long text — batch embed all chunks and store with shared sourceId
    const sourceId = crypto.randomUUID();
    const chunkTexts = chunks.map((c) => c.text);
    const vectors = await this.embedder.embedBatch(chunkTexts);

    await this.client.upsert(this.collectionName, {
      points: chunks.map((chunk, i) => ({
        id: crypto.randomUUID(),
        vector: vectors[i]!,
        payload: {
          text: chunk.text,
          sessionId,
          role,
          timestamp: now + i,
          sourceId,
          chunkIndex: chunk.index,
          totalChunks: chunk.totalChunks,
        } satisfies MemoryPayload,
      })),
    });
  }

  // Add multiple messages at once. Each entry is chunked individually.
  // More efficient than calling add() in a loop because we batch the
  // embedding computation AND the Qdrant upsert.
  async addBatch(
    entries: { text: string; role: string }[],
    sessionId: string,
    chunkOptions?: ChunkOptions
  ): Promise<void> {
    await this.ensureCollection();

    const now = Date.now();

    // Chunk all entries and flatten into a single list for batch embedding
    const allPoints: {
      text: string;
      role: string;
      timestamp: number;
      sourceId?: string;
      chunkIndex?: number;
      totalChunks?: number;
    }[] = [];

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i]!;
      const chunks = chunkText(entry.text, chunkOptions);

      if (chunks.length === 1) {
        allPoints.push({
          text: entry.text,
          role: entry.role,
          timestamp: now + i,
        });
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

    const vectors = await this.embedder.embedBatch(
      allPoints.map((p) => p.text)
    );

    await this.client.upsert(this.collectionName, {
      points: allPoints.map((point, i) => ({
        id: crypto.randomUUID(),
        vector: vectors[i]!,
        payload: {
          text: point.text,
          sessionId,
          role: point.role,
          timestamp: point.timestamp,
          ...(point.sourceId && {
            sourceId: point.sourceId,
            chunkIndex: point.chunkIndex,
            totalChunks: point.totalChunks,
          }),
        } satisfies MemoryPayload,
      })),
    });
  }

  // Search memory by semantic similarity, optionally blended with recency.
  //
  // Time-weighted scoring formula:
  //   finalScore = cosineScore × alpha + recencyScore × (1 - alpha)
  //
  // Over-fetch strategy: we request limit×3 from Qdrant (which only ranks
  // by cosine), compute blended scores, re-sort, and return top `limit`.
  // This ensures recent-but-moderately-similar results aren't missed.
  async search(
    query: string,
    options: SearchOptions = {}
  ): Promise<MemoryResult[]> {
    await this.ensureCollection();

    const {
      limit = 5,
      sessionId,
      timeWeight = true,
      alpha = 0.7,
      decayRate = 24,
    } = options;

    const vector = await this.embedder.embedQuery(query);

    const filter = sessionId
      ? { must: [{ key: "sessionId", match: { value: sessionId } }] }
      : undefined;

    // Over-fetch when time-weighting so we can re-rank
    const fetchLimit = timeWeight ? limit * 3 : limit;

    const results = await this.client.search(this.collectionName, {
      vector,
      limit: fetchLimit,
      filter,
    });

    const now = Date.now();

    const scored: MemoryResult[] = results.map((r) => {
      const cosineScore = r.score;
      const timestamp = r.payload?.timestamp as number;

      // Exponential decay: e^(-ageInHours / decayRate)
      const ageInHours = (now - timestamp) / 3_600_000;
      const recencyScore = Math.exp(-ageInHours / decayRate);

      const score = timeWeight
        ? cosineScore * alpha + recencyScore * (1 - alpha)
        : cosineScore;

      return {
        text: r.payload?.text as string,
        role: r.payload?.role as string,
        score,
        cosineScore,
        recencyScore,
        timestamp,
      };
    });

    // Re-sort by blended score and return top `limit`
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  // Delete all memories for a session. Useful for cleanup.
  async clearSession(sessionId: string): Promise<void> {
    await this.ensureCollection();

    await this.client.delete(this.collectionName, {
      filter: {
        must: [{ key: "sessionId", match: { value: sessionId } }],
      },
    });
  }
}
