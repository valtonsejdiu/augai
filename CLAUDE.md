# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

augai is a session memory system built on Qdrant (vector DB) with local embeddings via fastembed. It stores conversation messages as vectors and retrieves them by semantic similarity, with support for text chunking, time-weighted scoring, and multiple embedding models.

## Commands

```bash
npm start           # Run demo script (src/index.ts) — requires Qdrant on localhost:6333
npm run dev         # Watch mode for demo script
npm run cli         # Interactive conversation CLI with /search, /history, /clear, /quit
npm run compare     # Compare embedding models in-memory (no Qdrant needed)
npm run build       # TypeScript compilation to dist/
npx tsc --noEmit    # Type-check without emitting
```

## Prerequisites

Qdrant must be running locally on port 6333 for `start`, `dev`, and `cli` commands. The `compare` script runs purely in-memory.

## Architecture

**ESM TypeScript project** — uses `"type": "module"` in package.json, `"module": "nodenext"` in tsconfig. All imports use `.js` extensions (required for ESM resolution with TypeScript).

### Core Pipeline

`Embedder` (fastembed wrapper) → `SessionMemory` (Qdrant CRUD + search) → consumers (demo, CLI)

- **`src/embeddings.ts`** — `Embedder` class wraps fastembed's async init and AsyncGenerator API. Configurable model via `EmbeddingModel` enum. Each model has a fixed vector dimension (384, 768, or 1024) — different dimensions require separate Qdrant collections.

- **`src/memory.ts`** — `SessionMemory` class provides `add()`, `addBatch()`, `search()`, `clearSession()`. Takes `QdrantClient` and `Embedder` via constructor injection. Auto-creates Qdrant collections on first use. Long text is auto-chunked (via `chunker.ts`) with chunks linked by `sourceId`. Search uses time-weighted scoring by default: over-fetches 3× from Qdrant, blends cosine similarity with exponential recency decay, then re-ranks.

- **`src/chunker.ts`** — Pure function `chunkText()` that splits text on sentence boundaries with configurable overlap. No side effects.

- **`src/cli.ts`** — Interactive readline-based CLI. Each run gets a unique session ID.

- **`src/compare-models.ts`** — Standalone script comparing embedding models by computing cosine similarity in-memory without Qdrant.

## Key Patterns

- fastembed returns `Float32Array`; convert with `Array.from()` for Qdrant compatibility
- Qdrant point IDs use `crypto.randomUUID()`
- `MemoryPayload` is the metadata shape stored alongside each vector in Qdrant
- The `satisfies` keyword is used for payload type checking at upsert sites
