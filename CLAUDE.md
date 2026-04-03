# CLAUDE.md

## Project

augai — session memory system on Qdrant with pluggable embeddings (local fastembed, Ollama, cloud APIs, Rust sidecar). Supports hybrid dense+sparse search, cross-encoder reranking, text chunking, and time-weighted scoring. See README.md for full documentation.

## Commands

```bash
bun start            # Demo (src/index.ts) — needs Qdrant on localhost:6333
bun run cli          # Interactive CLI
bun run compare      # Compare embedding models in-memory (no Qdrant)
bun run compare:hybrid  # Hybrid vs dense benchmark (needs Qdrant + sidecar)
bun run typecheck    # tsc --noEmit
bun run infra:up     # docker-compose: Qdrant + Ollama + Infinity + sidecar
```

## Stack

- **Runtime:** Bun
- **ESM + TypeScript:** `module: "preserve"`, `moduleResolution: "bundler"`, strict mode
- **Deps:** `@qdrant/js-client-rest`, `openai`, `cohere-ai`, `voyageai`; `fastembed`, `unpdf` in `optionalDependencies`
- **No linter, no formatter, no tests**

## Architecture

`DenseEmbedder` (+ optional `SparseEmbedder` + `Reranker`) → `SessionMemory` → consumers

Backend selected via `AUGAI_EMBEDDER` env var (default: `"ollama"`). Reranker via `AUGAI_RERANKER`.

| File | Role |
|------|------|
| `src/types.ts` | Universal interfaces: `DenseEmbedder`, `SparseEmbedder`, `SparseVector`, `Reranker`, `RerankResult` |
| `src/provider.ts` | `createEmbedder()` factory — `local\|ollama\|openai\|cohere\|voyage\|mistral\|rust` |
| `src/reranker-provider.ts` | `createReranker()` factory — `infinity\|cohere\|voyage\|local\|rust` |
| `src/embeddings.ts` | `FastEmbedEmbedder` — fastembed wrapper (optional dep) |
| `src/embeddings-ollama.ts` | `OllamaEmbedder` — HTTP, auto-dim probe, batch=128 |
| `src/embeddings-{openai,cohere,voyage}.ts` | Cloud embedders |
| `src/embeddings-mistral.ts` | `MistralEmbedder` — Mistral API (`codestral-embed`, `mistral-embed`), built-in 1 req/s throttle for free tier |
| `src/sidecar-client.ts` | `SidecarDenseEmbedder` / `SidecarSparseEmbedder` / `SidecarReranker` — HTTP to Rust sidecar on :8081 |
| `src/rerank-infinity.ts` | `InfinityReranker` |
| `src/reranker-{cohere,voyage,local}.ts` | Cloud + BM25 rerankers |
| `src/memory.ts` | `SessionMemory` — `add()`, `addBatch()`, `addDocument()`, `search()`, `listSession()`, `listDocuments()`, `findByFileHash()`, `clearSession()`. Named-vector collections. Hybrid RRF search when sparse embedder present. Sigmoid rerank blending. Time-weighted scoring. |
| `src/ingest.ts` | `ingestFile()` + `resolveIngestPaths()` — file reading, PDF extraction (unpdf), SHA-256 dedup, format detection |
| `src/chunker.ts` | `chunkText()` sync + `chunkTextAsync()` for structured formats (markdown, HTML, LaTeX, code) |
| `src/chunk-text.ts` | Recursive sentence-aware text splitter with abbreviation handling and overlap |
| `src/chunk-structured.ts` | LangChain-backed splitter for structured formats (async, lazy-loaded) |
| `src/cli.ts` | Readline CLI, sidecar auto-detect. Commands: `/search`, `/history`, `/clear`, `/ingest`, `/docs`, `/quit` |
| `src/index.ts` | Demo script, sidecar auto-detect |
| `augai-embed/` | Rust HTTP sidecar — axum + fastembed-rs, exposes dense/sparse/rerank on :8081 |

## Code Rules

- **Performance is king** — optimize hot paths, avoid unnecessary allocations, prefer performant patterns
- **No comments unless essential** — only comment non-obvious logic, edge cases, or "why" (never "what")
- **No over-engineering** — no abstractions for single-use code, no speculative features
- **Minimal diffs** — change only what's needed. Don't refactor, reformat, or annotate untouched code

## Conventions

- `Float32Array` → `Array.from()` for Qdrant compatibility
- Point IDs: `crypto.randomUUID()`
- Payload type safety: `satisfies MemoryPayload` at upsert sites
- `MemoryPayload`: `{ text, sessionId, role, timestamp, sourceId?, chunkIndex?, totalChunks?, sourceFile?, sourceType?, fileHash? }`
- All embedder implementations satisfy `DenseEmbedder` from `./types.js`
- Rerank scores sigmoid-normalized before blending: `1 / (1 + Math.exp(-relevanceScore))`
- Over-fetch: `min(limit × 10, 1000)` for reranking; `min(limit × 3, 1000)` for time-weight only

## Qdrant Schema

Named vectors always — `{ vectors: { dense: { size, distance: "Cosine" } }, sparse_vectors?: { sparse: {} } }`. Old flat-vector collections are incompatible. Drop or set `AUGAI_COLLECTION=session_memory_v2`.

## Known Qdrant JS Client Footguns

- **`client.upsert()` named vectors**: use `vector: { dense: [...] }` (singular key) — JS client diverges from REST spec
- **`client.query()` vs `client.search()`**: `query()` wraps results as `{ points: [] }`; `search()` returns array directly — always access `.points` after `query()`
