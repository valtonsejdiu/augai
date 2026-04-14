import { QdrantClient } from "@qdrant/js-client-rest";
import { createEmbedder } from "../provider.js";
import { createReranker } from "../reranker-provider.js";
import { SessionMemory } from "../memory.js";
import type { DenseEmbedder } from "../types.js";

let _client: QdrantClient | null = null;
let _embedder: DenseEmbedder | null = null;
let _memory: SessionMemory | null = null;

export function getClient(): QdrantClient {
  if (!_client) {
    _client = new QdrantClient({ url: process.env.QDRANT_URL ?? "http://localhost:6333" });
  }
  return _client;
}

export async function getDenseEmbedder(): Promise<DenseEmbedder> {
  if (!_embedder) _embedder = await createEmbedder();
  return _embedder;
}

export async function getMemory(): Promise<SessionMemory> {
  if (_memory) return _memory;

  const dense = await getDenseEmbedder();

  let sparse = null;
  let reranker = null;
  try {
    sparse = new (await import("../sidecar-client.js")).SidecarSparseEmbedder();
  } catch (e) {
    process.stderr.write(`[augai] sparse embedder unavailable: ${e}\n`);
  }
  try {
    reranker = await createReranker();
  } catch (e) {
    process.stderr.write(`[augai] reranker unavailable: ${e}\n`);
  }

  _memory = new SessionMemory(getClient(), dense, undefined, sparse, reranker);
  return _memory;
}
