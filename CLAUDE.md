# CLAUDE.md

## Project

augai — session memory system on Qdrant (vector DB) with local embeddings (fastembed). Stores conversation messages as vectors, retrieves by semantic similarity. Supports text chunking, time-weighted scoring, multiple embedding models.

## Commands

```bash
npm start           # Demo script (src/index.ts) — needs Qdrant on localhost:6333
npm run dev         # Watch mode for demo
npm run cli         # Interactive CLI (/search, /history, /clear, /quit) — needs Qdrant
npm run compare     # Compare embedding models in-memory (no Qdrant)
npm run build       # tsc → dist/
npx tsc --noEmit    # Type-check only
```

## Stack

- **Runtime:** Node.js with `tsx` (no build step needed for dev)
- **ESM + TypeScript:** `"type": "module"`, `"module": "nodenext"`, strict mode. All imports use `.js` extensions
- **Deps:** `@qdrant/js-client-rest`, `fastembed`
- **Dev deps:** `typescript`, `tsx`, `@types/node`
- **No linter, no formatter, no tests configured**

## Architecture

`Embedder` → `SessionMemory` → consumers (demo, CLI)

| File | Role |
|------|------|
| `src/embeddings.ts` | `Embedder` class — fastembed wrapper, lazy init, `embedQuery()` / `embedBatch()`. Models: 384/768/1024-dim (different dims = separate collections) |
| `src/memory.ts` | `SessionMemory` — `add()`, `addBatch()`, `search()`, `clearSession()`. Constructor injection (QdrantClient + Embedder). Auto-creates collections. Auto-chunks long text via chunker, links chunks by `sourceId`. Time-weighted search: over-fetch 3×, blend cosine + recency decay, re-rank |
| `src/chunker.ts` | `chunkText()` — pure function, sentence-boundary splitting with overlap |
| `src/cli.ts` | Readline CLI, unique session per run |
| `src/compare-models.ts` | In-memory model comparison via cosine similarity |
| `src/index.ts` | Demo script — batch insert + search examples |

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
