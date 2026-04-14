# augai

Session memory system built on [Qdrant](https://qdrant.tech) with pluggable embedding backends. Stores conversation messages as vectors, retrieves by semantic similarity. Supports text chunking, time-weighted scoring, hybrid dense+sparse search, cross-encoder reranking, and document ingestion (PDF, HTML, Markdown, plain text) with SHA-256 deduplication.

## Backends

| `AUGAI_EMBEDDER` | Description | Hybrid search | Reranking |
|-----------------|-------------|---------------|-----------|
| `ollama` (default) | Local via [Ollama](https://ollama.com) — `nomic-embed-text` | no | optional |
| `local` | Local via fastembed (ONNX) | no | optional |
| `openai` | OpenAI Embeddings API | no | optional |
| `cohere` | Cohere Embed API (asymmetric) | no | optional |
| `voyage` | Voyage AI (VoyageStrategy: 3-model index/query/fallback) | no | optional |
| `mistral` | Mistral API (`codestral-embed`, `mistral-embed`) — 1 req/s throttle for free tier | no | optional |
| `rust` | Rust sidecar (`augai-embed`) — NomicEmbedTextV15 + SPLADE sparse | yes | yes (BGE) |

## Prerequisites

- [Bun](https://bun.sh) ≥ 1.0
- [Docker](https://www.docker.com) + Compose (for infrastructure)
- [Rust toolchain](https://rustup.rs) (only if building the sidecar locally)

## Quick Start

```bash
# 1. Install dependencies
bun install

# 2. Start infrastructure (Qdrant + Ollama + Infinity reranker)
bun run infra:up

# 3. Run interactive CLI
bun run cli

# CLI commands: /search <query>, /history, /clear, /ingest <path>, /docs, /quit
```

## Commands

```bash
bun start                      # Demo script
bun run dev                    # Watch mode
bun run cli                    # Interactive CLI (default: ollama backend)
bun run cli:openai             # CLI with OpenAI backend
bun run cli:cohere             # CLI with Cohere backend
bun run cli:voyage             # CLI with Voyage backend
bun run cli:rerank             # CLI with Infinity reranker
bun run cli:ollama:mxbai       # CLI with mxbai-embed-large model
bun run compare                # Compare embedding models in-memory
bun run compare:hybrid         # Hybrid vs dense search benchmark (needs sidecar)
bun run typecheck              # TypeScript type-check only

# Infrastructure
bun run infra:up               # Start Qdrant + Ollama + Infinity + sidecar
bun run infra:down             # Stop all services
bun run infra:pull             # Pull Ollama model

# Rust sidecar (optional — enables hybrid search)
bun run sidecar:build          # cargo build --release
bun run sidecar:start          # Start sidecar on :8081
```

## Hybrid Search (Rust Sidecar)

The Rust sidecar (`augai-embed/`) provides hardware-accelerated dense + SPLADE sparse embeddings and BGE cross-encoder reranking. It is an optional HTTP service on `:8081` — the CLI and demo auto-detect it via `GET /health` and fall back to dense-only if unreachable.

```bash
# Apple Silicon — GPU acceleration
AUGAI_EMBED_FEATURES=metal bun run infra:up

# Apple Silicon — CPU BLAS
AUGAI_EMBED_FEATURES=accelerate bun run infra:up

# Or build manually
bun run sidecar:build
AUGAI_EMBEDDER=rust bun run cli
```

The sidecar uses:
- **NomicEmbedTextV15** (768d, 8K context) for dense embeddings
- **SPLADE-PP-v1** for sparse embeddings
- **BGERerankerV2M3** for cross-encoder reranking

## Document Ingestion

Ingest PDF, HTML, Markdown, and plain text files directly into session memory via the CLI. Documents are chunked with format-aware splitting, embedded, and stored in Qdrant. SHA-256 hashing prevents duplicate ingestion.

```bash
# Ingest a single file
you> /ingest ./docs/paper.pdf
[done] paper.pdf — 42 chunks (1230ms)

# Ingest multiple files with glob
you> /ingest ./docs/*.md
[done] README.md — 3 chunks (450ms)
[skip] notes.md (already ingested)

# List ingested documents
you> /docs
  [17:30:00] paper.pdf (pdf) — 42 chunks
  [17:30:01] README.md (md) — 3 chunks

# Search across ingested documents
you> /search database connection
  [score=0.8234 ...] (document) ...
```

**Supported formats:** `.pdf`, `.html`, `.htm`, `.md`, `.txt`

PDF support requires the optional `unpdf` dependency:
```bash
bun add unpdf
```

**Persistence:** Qdrant's `qdrant_data` Docker volume persists all ingested documents across container restarts. No additional configuration needed.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUGAI_EMBEDDER` | `ollama` | Embedding backend: `local\|ollama\|openai\|cohere\|voyage\|mistral\|rust` |
| `AUGAI_RERANKER` | unset | Reranker: `infinity\|cohere\|voyage\|local\|rust` (disabled if unset) |
| `AUGAI_COLLECTION` | `session_memory` | Qdrant collection name |
| `AUGAI_EMBED_URL` | `http://localhost:8081` | Rust sidecar endpoint |
| `AUGAI_EMBED_FEATURES` | — | Sidecar build features: `metal` or `accelerate` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `nomic-embed-text` | Ollama model name |
| `INFINITY_URL` | `http://localhost:7997` | Infinity reranker endpoint |
| `INFINITY_RERANK_MODEL` | `mixedbread-ai/mxbai-rerank-xsmall-v1` | Infinity rerank model |
| `OPENAI_API_KEY` | — | Required for `openai` backend |
| `CO_API_KEY` | — | Required for `cohere` backend/reranker |
| `VOYAGEAI_API_KEY` | — | Required for `voyage` backend/reranker |
| `MISTRAL_API_KEY` | — | Required for `mistral` backend |
| `VOYAGE_INDEX_MODEL` | `voyage-4-large` | Voyage model for indexing documents |
| `VOYAGE_QUERY_MODEL` | `voyage-4-lite` | Voyage model for queries |
| `VOYAGE_FALLBACK_MODEL` | `voyage-4` | Voyage fallback model on rate-limit |

## Architecture

```
DenseEmbedder (+ optional SparseEmbedder + Reranker)
       ↓
 SessionMemory
       ↓
consumers: CLI, demo, compare scripts
```

| File | Role |
|------|------|
| `src/types.ts` | Interfaces: `DenseEmbedder`, `SparseEmbedder`, `Reranker`, `RerankResult`, `SparseVector` |
| `src/provider.ts` | `createEmbedder()` — factory driven by `AUGAI_EMBEDDER` |
| `src/reranker-provider.ts` | `createReranker()` — factory driven by `AUGAI_RERANKER` |
| `src/embeddings.ts` | `FastEmbedEmbedder` — local fastembed (optional dep) |
| `src/embeddings-ollama.ts` | `OllamaEmbedder` — HTTP, auto-dim probe, batch=128 |
| `src/embeddings-openai.ts` | `OpenAIEmbedder` |
| `src/embeddings-cohere.ts` | `CohereEmbedder` — asymmetric `input_type` |
| `src/embeddings-voyage.ts` | `VoyageEmbedder` — VoyageStrategy + retry/fallback |
| `src/embeddings-mistral.ts` | `MistralEmbedder` — `codestral-embed` / `mistral-embed`, 1 req/s throttle |
| `src/sidecar-client.ts` | `SidecarDenseEmbedder`, `SidecarSparseEmbedder`, `SidecarReranker` |
| `src/rerank-infinity.ts` | `InfinityReranker` |
| `src/reranker-cohere.ts` | `CohereReranker` |
| `src/reranker-voyage.ts` | `VoyageReranker` |
| `src/reranker-local.ts` | `LocalReranker` — BM25, fully offline |
| `src/memory.ts` | `SessionMemory` — add, addDocument, search, hybrid RRF, reranking, time-weight, dedup |
| `src/ingest.ts` | Document ingestion — PDF/HTML/MD/TXT reading, SHA-256 dedup, format detection |
| `src/chunker.ts` | `chunkText()` — sentence-boundary splitting with overlap |
| `src/chunk-text.ts` | Recursive sentence-aware text splitter with abbreviation handling |
| `src/chunk-structured.ts` | LangChain-backed splitter for structured formats (markdown, HTML, LaTeX, code) |
| `src/cli.ts` | Interactive CLI with `/search`, `/ingest`, `/docs`, `/history`, `/clear` |
| `src/index.ts` | Demo script |
| `src/compare-models.ts` | In-memory model comparison |
| `src/compare-hybrid.ts` | Hybrid vs dense search benchmark |
| `augai-embed/` | Rust HTTP sidecar (axum + fastembed-rs) |

## Infrastructure (Docker Compose)

| Service | Port | Purpose |
|---------|------|---------|
| `qdrant` | 6333 | Vector database |
| `ollama` | 11434 | Local LLM/embedding server |
| `infinity` | 7997 | Cross-encoder reranker (`mxbai-rerank-xsmall-v1`) |
| `augai-embed` | 8081 | Rust sidecar (dense + sparse + rerank) |

## Qdrant Schema

Collections use **named vectors**: `{ dense: { size, distance: "Cosine" } }` + optional `sparse_vectors`. Old flat-vector `session_memory` collections from pre-merge are incompatible — drop the collection or set `AUGAI_COLLECTION=session_memory_v2` when upgrading.
