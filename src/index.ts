import { QdrantClient } from "@qdrant/js-client-rest";
import type { SparseEmbedder } from "./types.js";
import { createEmbedder } from "./provider.js";
import { createReranker } from "./reranker-provider.js";
import { SessionMemory } from "./memory.js";

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

  const SESSION_ID = "demo-session-001";

  console.log("Storing conversation in memory...\n");

  await memory.addBatch(
    [
      { text: "How do I connect to a PostgreSQL database from Node.js?", role: "user" },
      { text: "You can use the 'pg' package. Install it with npm install pg, then create a Pool with your connection string.", role: "assistant" },
      { text: "What about connection pooling? Is it built in?", role: "user" },
      { text: "Yes, pg.Pool manages a pool of connections automatically. Set max, idleTimeoutMillis, and connectionTimeoutMillis to tune it.", role: "assistant" },
      { text: "I'm also working on a React frontend with Tailwind CSS", role: "user" },
      { text: "Great combo. Make sure to configure your tailwind.config.js content paths to include your component files.", role: "assistant" },
      { text: "My Docker containers keep running out of memory", role: "user" },
      { text: "Set memory limits in your docker-compose.yml with deploy.resources.limits.memory. Also check for memory leaks in your app.", role: "assistant" },
      { text: "How do I handle database migrations?", role: "user" },
      { text: "Use a migration tool like knex, prisma migrate, or golang-migrate. They version your schema changes and apply them in order.", role: "assistant" },
    ],
    SESSION_ID
  );

  console.log("Stored 10 messages. Now searching by meaning...\n");

  const queries = [
    "database connection setup",
    "CSS styling framework",
    "container resource limits",
    "schema versioning",
  ];

  for (const query of queries) {
    const results = await memory.search(query, { limit: 2, sessionId: SESSION_ID });

    console.log(`Query: "${query}"`);
    for (const r of results) {
      const rrk = r.rerankScore !== undefined ? ` rrk=${r.rerankScore.toFixed(4)}` : "";
      console.log(
        `  [score=${r.score.toFixed(4)} cos=${r.cosineScore.toFixed(4)}${rrk} rec=${r.recencyScore.toFixed(4)}] (${r.role}) ${r.text.substring(0, 80)}...`
      );
    }
    console.log();
  }

  await memory.clearSession(SESSION_ID);
  console.log("Session cleared.");
}

main().catch(console.error);
