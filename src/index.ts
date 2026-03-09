import { QdrantClient } from "@qdrant/js-client-rest";
import { Embedder } from "./embeddings.js";
import { SessionMemory } from "./memory.js";

async function main() {
  // --- SETUP ---
  const client = new QdrantClient({ host: "localhost", port: 6333 });
  const embedder = new Embedder();
  const memory = new SessionMemory(client, embedder);

  const SESSION_ID = "demo-session-001";

  // --- SIMULATE A CONVERSATION ---
  // Imagine a user having a technical discussion. We store each message.
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

  // --- SEARCH: the power of semantic memory ---
  // Notice: the search queries DON'T match the stored text word-for-word.
  // The embedding model understands MEANING, not just keywords.

  const queries = [
    "database connection setup",    // Should find the PostgreSQL discussion
    "CSS styling framework",        // Should find the Tailwind conversation
    "container resource limits",    // Should find the Docker memory discussion
    "schema versioning",            // Should find the migrations discussion
  ];

  for (const query of queries) {
    const results = await memory.search(query, {
      limit: 2,
      sessionId: SESSION_ID,
    });

    console.log(`Query: "${query}"`);
    for (const r of results) {
      console.log(`  [score=${r.score.toFixed(4)} cos=${r.cosineScore.toFixed(4)} rec=${r.recencyScore.toFixed(4)}] (${r.role}) ${r.text.substring(0, 80)}...`);
    }
    console.log();
  }

  // --- CLEANUP ---
  // In production you'd keep the data. Here we clean up so the demo
  // is re-runnable without duplicates.
  await memory.clearSession(SESSION_ID);
  console.log("Session cleared.");
}

main().catch(console.error);
