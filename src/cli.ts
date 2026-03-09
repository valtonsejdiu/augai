import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { QdrantClient } from "@qdrant/js-client-rest";
import { Embedder } from "./embeddings.js";
import { SessionMemory } from "./memory.js";

// Interactive conversation loop with semantic memory.
// Type messages to store them, use commands to search and manage.

const SESSION_ID = `cli-${Date.now()}`;

async function main() {
  const client = new QdrantClient({ host: "localhost", port: 6333 });
  const embedder = new Embedder();
  const memory = new SessionMemory(client, embedder);

  const rl = readline.createInterface({ input: stdin, output: stdout });

  console.log("augai — Session Memory CLI");
  console.log(`Session: ${SESSION_ID}`);
  console.log("Commands: /search <query>, /history, /clear, /quit");
  console.log("Anything else is stored as a user message.\n");

  let messageCount = 0;

  while (true) {
    const input = await rl.question("you> ").catch(() => null);

    // EOF (Ctrl+D)
    if (input === null) break;

    const trimmed = input.trim();
    if (!trimmed) continue;

    // --- Commands ---

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
      const results = await memory.search("", {
        limit: 20,
        sessionId: SESSION_ID,
        timeWeight: false,
      });
      // Sort by timestamp ascending for chronological display
      results.sort((a, b) => a.timestamp - b.timestamp);

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

      const results = await memory.search(query, {
        limit: 5,
        sessionId: SESSION_ID,
      });

      if (results.length === 0) {
        console.log("No results found.\n");
      } else {
        console.log(`\n--- Search: "${query}" ---`);
        for (const r of results) {
          const time = new Date(r.timestamp).toLocaleTimeString();
          console.log(
            `  [score=${r.score.toFixed(4)} cos=${r.cosineScore.toFixed(4)} rec=${r.recencyScore.toFixed(4)}] [${time}] (${r.role})`
          );
          console.log(`    ${r.text.substring(0, 120)}`);
        }
        console.log();
      }
      continue;
    }

    if (trimmed.startsWith("/")) {
      console.log(`Unknown command: ${trimmed.split(" ")[0]}`);
      console.log("Commands: /search <query>, /history, /clear, /quit\n");
      continue;
    }

    // --- Store user message ---
    await memory.add(trimmed, SESSION_ID, "user");
    messageCount++;
    console.log(`Stored (${messageCount} messages in session).\n`);
  }

  rl.close();
}

main().catch(console.error);
