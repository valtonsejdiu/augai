# CLAUDE.md

## Project

augai — session memory system on Qdrant (vector DB) with pluggable embeddings backends (local fastembed, Ollama, cloud APIs, or Rust sidecar). Stores conversation messages as vectors, retrieves by semantic similarity. Supports text chunking, time-weighted scoring, hybrid dense+sparse search, and cross-encoder reranking.

## Commands

```bash
bun start                # Demo script (src/index.ts) — needs Qdrant on localhost:6333
bun run dev              # Watch mode for demo
bun run cli              # Interactive CLI (/search, /history, /clear, /quit) — needs Qdrant
bun run compare          # Compare embedding models in-memory (no Qdrant)
bun run compare:hybrid   # Compare hybrid vs dense search (needs Qdrant + sidecar)
bun run build            # tsc → dist/
bunx tsc --noEmit        # Type-check only

# Rust sidecar (optional — enables hybrid search)
bun run sidecar:build    # cargo build --release (requires Rust toolchain)
bun run sidecar:start    # start sidecar on :8081

# Infrastructure
bun run infra:up         # docker-compose up -d (Qdrant + Ollama + Infinity + sidecar)
bun run infra:down       # docker-compose down
bun run infra:pull       # docker-compose pull
```

## Stack

- **Runtime:** Bun (replaces Node.js/tsx)
- **ESM + TypeScript:** `"type": "module"`, `module: "preserve"`, `moduleResolution: "bundler"`, strict mode
- **Core deps:** `@qdrant/js-client-rest`, `openai`, `cohere-ai`, `voyageai`
- **Optional deps:** `fastembed` (Bun ONNX incompatibility — install only if needed)
- **No linter, no formatter, no tests configured**

## Architecture

`DenseEmbedder` (+ optional `SparseEmbedder` + `Reranker`) → `SessionMemory` → consumers (demo, CLI)

Backend selected via `AUGAI_EMBEDDER` env var (default: `"ollama"`).

| File | Role |
|------|------|
| `src/types.ts` | Universal interfaces: `DenseEmbedder`, `SparseEmbedder`, `SparseVector`, `Reranker`, `RerankResult`, `ChunkOptions`, `Chunk` |
| `src/provider.ts` | `createEmbedder()` factory — env-var driven (`AUGAI_EMBEDDER`): `local\|ollama\|openai\|cohere\|voyage\|rust` |
| `src/reranker-provider.ts` | `createReranker()` factory — env-var driven (`AUGAI_RERANKER`): `infinity\|cohere\|voyage\|local\|rust` |
| `src/embeddings.ts` | `FastEmbedEmbedder` — fastembed wrapper (optional dep) |
| `src/embeddings-ollama.ts` | `OllamaEmbedder` — Ollama HTTP, auto-dim probe, batch=128 |
| `src/embeddings-openai.ts` | `OpenAIEmbedder` — OpenAI Embeddings API |
| `src/embeddings-cohere.ts` | `CohereEmbedder` — asymmetric `input_type` (search_query vs search_document) |
| `src/embeddings-voyage.ts` | `VoyageEmbedder` — VoyageStrategy (3 models: index/query/fallback), `isRetryable()` + fallback |
| `src/sidecar-client.ts` | `SidecarDenseEmbedder`, `SidecarSparseEmbedder`, `SidecarReranker` — HTTP client for Rust sidecar on :8081 |
| `src/rerank-infinity.ts` | `InfinityReranker` — Infinity cross-encoder HTTP |
| `src/reranker-cohere.ts` | `CohereReranker` |
| `src/reranker-voyage.ts` | `VoyageReranker` |
| `src/reranker-local.ts` | `LocalReranker` — BM25, offline, no external deps |
| `src/memory.ts` | `SessionMemory` — `add()`, `addBatch()`, `search()`, `listSession()`, `clearSession()`. Named-vector Qdrant collections. Hybrid search (RRF) when sparse embedder present. Sigmoid-normalized rerank blending. Time-weighted scoring. |
| `src/chunker.ts` | `chunkText()` — sentence-boundary splitting with overlap |
| `src/cli.ts` | Readline CLI, sidecar auto-detect, unique session per run |
| `src/index.ts` | Demo script — batch insert + search examples, sidecar auto-detect |
| `src/compare-models.ts` | In-memory embedding model comparison |
| `src/compare-hybrid.ts` | Hybrid vs dense search comparison (needs Qdrant + sidecar) |
| `augai-embed/` | Rust HTTP sidecar — axum + fastembed-rs, exposes dense/sparse/rerank on :8081 |

## Running Modes

| Mode | `AUGAI_EMBEDDER` | Sidecar | Hybrid search | Reranking |
|------|-----------------|---------|---------------|-----------|
| Local (fastembed) | `local` | no | no | optional |
| Ollama | `ollama` (default) | no | no | optional |
| Cloud | `openai\|cohere\|voyage` | no | no | optional |
| Rust sidecar | `rust` | required | yes (SPLADE) | yes (BGE) |

Sidecar is auto-detected via `GET /health` — falls back to dense-only if unreachable.

## Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `AUGAI_EMBEDDER` | `ollama` | Backend: `local\|ollama\|openai\|cohere\|voyage\|rust` |
| `AUGAI_RERANKER` | unset | Reranker: `infinity\|cohere\|voyage\|local\|rust` |
| `AUGAI_COLLECTION` | `session_memory` | Qdrant collection name |
| `AUGAI_EMBED_URL` | `http://localhost:8081` | Rust sidecar endpoint |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `nomic-embed-text` | Ollama model |
| `INFINITY_URL` | `http://localhost:7997` | Infinity reranker endpoint |
| `OPENAI_API_KEY` | — | Required for `openai` backend |
| `CO_API_KEY` | — | Required for `cohere` backend/reranker |
| `VOYAGEAI_API_KEY` | — | Required for `voyage` backend/reranker |
| `VOYAGE_INDEX_MODEL` | `voyage-4-large` | Voyage model for indexing |
| `VOYAGE_QUERY_MODEL` | `voyage-4-lite` | Voyage model for queries |
| `VOYAGE_FALLBACK_MODEL` | `voyage-4` | Voyage fallback model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |

## Code Rules

- **Performance is king** — optimize hot paths, avoid unnecessary allocations, prefer performant patterns. Readability serves performance, never the reverse
- **No comments unless essential** — if the code is self-explanatory, no comment. Only comment non-obvious logic, edge cases, or "why" (never "what")
- **No over-engineering** — no abstractions for single-use code, no speculative features, no unnecessary error handling for impossible states
- **Minimal diffs** — change only what's needed. Don't refactor, reformat, or annotate untouched code

## Conventions

- `Float32Array` → `Array.from()` for Qdrant compatibility
- Point IDs: `crypto.randomUUID()`
- Payload type safety: `satisfies MemoryPayload` at upsert sites
- `MemoryPayload`: `{ text, sessionId, role, timestamp, sourceId?, chunkIndex?, totalChunks? }`
- All embedder implementations must satisfy `DenseEmbedder` from `./types.js`
- Rerank scores sigmoid-normalized before blending: `1 / (1 + Math.exp(-relevanceScore))`

## Qdrant Schema (named vectors — breaking change from pre-merge)

Collections always use named vectors:
```typescript
vectors: { dense: { size: embedder.dimension, distance: "Cosine" } },
sparse_vectors: sparse ? { sparse: { index: { type: "plain" } } } : undefined
```
**Old flat-vector `session_memory` collections are incompatible.** Drop the old collection or set `AUGAI_COLLECTION=session_memory_v2` when upgrading.

## Known Qdrant JS Client Footguns

- **`client.upsert()` named vectors**: use `vector: { dense: [...] }` (singular key), NOT `vectors:` — the JS client diverges from the REST spec here
- **`client.query()` vs `client.search()`**: `query()` wraps results as `{ points: [] }`; `search()` returns the array directly — always access `.points` after `query()`
