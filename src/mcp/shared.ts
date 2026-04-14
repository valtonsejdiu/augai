import { QdrantClient } from "@qdrant/js-client-rest";
import { createEmbedder } from "../provider.js";
import { createReranker } from "../reranker-provider.js";
import { SessionMemory } from "../memory.js";
import type { DenseEmbedder } from "../types.js";

let _client: QdrantClient | null = null;
let _embedderPromise: Promise<DenseEmbedder> | null = null;
let _memoryPromise: Promise<SessionMemory> | null = null;

export function getClient(): QdrantClient {
  if (!_client) {
    _client = new QdrantClient({ url: process.env.QDRANT_URL ?? "http://localhost:6333" });
  }
  return _client;
}

export function getDenseEmbedder(): Promise<DenseEmbedder> {
  if (!_embedderPromise) _embedderPromise = createEmbedder();
  return _embedderPromise;
}

async function _initMemory(): Promise<SessionMemory> {
  const dense = await getDenseEmbedder();

  let sparse = null;
  let reranker = null;
  try {
    const candidate = new (await import("../sidecar-client.js")).SidecarSparseEmbedder();
    await candidate.embedBatch(["probe"]);
    sparse = candidate;
  } catch (e) {
    process.stderr.write(`[augai] sparse embedder unavailable: ${e}\n`);
  }
  try {
    reranker = await createReranker();
  } catch (e) {
    process.stderr.write(`[augai] reranker unavailable: ${e}\n`);
  }

  return new SessionMemory(getClient(), dense, undefined, sparse, reranker);
}

export function getMemory(): Promise<SessionMemory> {
  if (!_memoryPromise) _memoryPromise = _initMemory();
  return _memoryPromise;
}
