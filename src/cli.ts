import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { basename } from "node:path";
import { QdrantClient } from "@qdrant/js-client-rest";
import type { SparseEmbedder } from "./types.js";
import { createEmbedder } from "./provider.js";
import { createReranker } from "./reranker-provider.js";
import { SessionMemory } from "./memory.js";

const SESSION_ID = `cli-${Date.now()}`;

async function checkSidecarHealth(): Promise<boolean> {
  const url = (process.env.AUGAI_EMBED_URL ?? "http://localhost:8081").replace(/\/$/, "");
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch {
    return false;
  }
}

async function main() {
  const client = new QdrantClient({ url: process.env.QDRANT_URL ?? "http://localhost:6333" });

  const sidecarUp = await checkSidecarHealth();
  let sparse: SparseEmbedder | null = null;
  if (sidecarUp) {
    const { SidecarSparseEmbedder } = await import("./sidecar-client.js");
    sparse = new SidecarSparseEmbedder();
    console.log("[hybrid] Rust sidecar up — hybrid dense+sparse search enabled");
  }

  const embedder = await createEmbedder();
  console.log(`[embedder] ${process.env.AUGAI_EMBEDDER || "ollama"} (${embedder.dimension}d)`);

  const reranker = await createReranker();
  if (reranker) console.log(`[reranker] ${process.env.AUGAI_RERANKER}`);

  const memory = new SessionMemory(client, embedder, undefined, sparse, reranker);

  const rl = readline.createInterface({ input: stdin, output: stdout });

  console.log("augai — Session Memory CLI");
  console.log(`Session: ${SESSION_ID}`);
  console.log("Commands: /search <query>, /history, /clear, /ingest <path>, /docs, /quit");
  console.log("Anything else is stored as a user message.\n");

  let messageCount = 0;

  while (true) {
    const input = await rl.question("you> ").catch(() => null);

    if (input === null) break;

    const trimmed = input.trim();
    if (!trimmed) continue;

    if (trimmed === "/quit") {
      console.log("Goodbye.");
      break;
    }

    if (trimmed === "/clear") {
      await memory.clearSession(SESSION_ID);
      messageCount = 0;
      console.log("Session cleared.\n");
      continue;
    }

    if (trimmed === "/history") {
      const results = await memory.listSession(SESSION_ID, 20);

      if (results.length === 0) {
        console.log("No messages yet.\n");
      } else {
        console.log(`\n--- History (${results.length} messages) ---`);
        for (const r of results) {
          const time = new Date(r.timestamp).toLocaleTimeString();
          console.log(`  [${time}] (${r.role}) ${r.text.substring(0, 100)}`);
        }
        console.log();
      }
      continue;
    }

    if (trimmed.startsWith("/search ")) {
      const query = trimmed.slice(8).trim();
      if (!query) {
        console.log("Usage: /search <query>\n");
        continue;
      }

      const results = await memory.search(query, { limit: 5, sessionId: SESSION_ID });

      if (results.length === 0) {
        console.log("No results found.\n");
      } else {
        console.log(`\n--- Search: "${query}" ---`);
        for (const r of results) {
          const time = new Date(r.timestamp).toLocaleTimeString();
          const rrk = r.rerankScore !== undefined ? ` rrk=${r.rerankScore.toFixed(4)}` : "";
          const sprs = r.sparseScore !== undefined ? ` sprs=${r.sparseScore.toFixed(4)}` : "";
          console.log(
            `  [score=${r.score.toFixed(4)} cos=${r.cosineScore.toFixed(4)}${rrk}${sprs} rec=${r.recencyScore.toFixed(4)}] [${time}] (${r.role})`
          );
          console.log(`    ${r.text.substring(0, 120)}`);
        }
        console.log();
      }
      continue;
    }

    if (trimmed.startsWith("/ingest ")) {
      const pathInput = trimmed.slice(8).trim();
      if (!pathInput) {
        console.log("Usage: /ingest <path-or-glob>\n");
        continue;
      }
      const { resolveIngestPaths, ingestFile } = await import("./ingest.js");
      const paths = await resolveIngestPaths(pathInput);
      if (paths.length === 0) {
        console.log("No supported files found (supported: .pdf .html .htm .md .txt)\n");
        continue;
      }
      for (const p of paths) {
        try {
          const result = await ingestFile(p, memory, SESSION_ID);
          if (result.skipped) {
            console.log(`[skip] ${result.filename} (already ingested)`);
          } else {
            console.log(`[done] ${result.filename} — ${result.chunks} chunks (${result.timeMs}ms)`);
          }
        } catch (err) {
          console.log(`[fail] ${basename(p)} — ${err instanceof Error ? err.message : err}`);
        }
      }
      console.log();
      continue;
    }

    if (trimmed === "/docs") {
      const docs = await memory.listDocuments(SESSION_ID);
      if (docs.length === 0) {
        console.log("No documents ingested in this session.\n");
        continue;
      }
      console.log(`\n--- Ingested Documents (${docs.length}) ---`);
      for (const doc of docs) {
        const time = new Date(doc.timestamp).toLocaleTimeString();
        console.log(`  [${time}] ${doc.sourceFile} (${doc.sourceType}) — ${doc.chunks} chunks`);
      }
      console.log();
      continue;
    }

    if (trimmed.startsWith("/")) {
      console.log(`Unknown command: ${trimmed.split(" ")[0]}`);
      console.log("Commands: /search <query>, /history, /clear, /ingest <path>, /docs, /quit\n");
      continue;
    }

    await memory.add(trimmed, SESSION_ID, "user");
    messageCount++;
    console.log(`Stored (${messageCount} messages in session).\n`);
  }

  rl.close();
}

main().catch(console.error);
