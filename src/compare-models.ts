import { Embedder, EmbeddingModel } from "./embeddings.js";

// Compare how different embedding models rank the same texts.
// Operates purely in memory — no Qdrant needed.

// Cosine similarity between two vectors (manual dot product).
// Both vectors should already be normalized (fastembed does this),
// so dot product = cosine similarity.
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Test texts — a mix of topics to see how models differentiate them
const texts = [
  "PostgreSQL supports JSONB columns for storing semi-structured data efficiently.",
  "React hooks like useState and useEffect simplify component state management.",
  "Docker containers isolate applications and their dependencies from the host system.",
  "Machine learning models need training data that is representative and unbiased.",
  "REST APIs use HTTP methods like GET, POST, PUT, and DELETE for CRUD operations.",
  "CSS Grid and Flexbox are modern layout systems for responsive web design.",
  "Kubernetes orchestrates container deployment, scaling, and networking.",
  "Neural networks consist of layers of interconnected nodes that transform input data.",
];

// Queries — some direct, some requiring semantic understanding
const queries = [
  "database storage format",          // Should match PostgreSQL/JSONB
  "frontend state",                   // Should match React hooks
  "container orchestration",          // Should match Docker and/or Kubernetes
  "deep learning architecture",       // Should match neural networks and/or ML
];

// Models to compare. Start with two that share the same dimension for simplicity.
// Add BGEBaseENV15 (768-dim) if you want to see a larger model's behavior.
const modelsToCompare: { name: string; model: EmbeddingModel }[] = [
  { name: "AllMiniLML6V2 (384d)", model: EmbeddingModel.AllMiniLML6V2 },
  { name: "BGESmallENV15 (384d)", model: EmbeddingModel.BGESmallENV15 },
];

async function main() {
  for (const { name, model } of modelsToCompare) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`Model: ${name}`);
    console.log("=".repeat(60));

    const embedder = new Embedder(model);

    // Embed all texts and queries
    console.log("Embedding texts...");
    const textVectors = await embedder.embedBatch(texts);
    const queryVectors = await embedder.embedBatch(queries);

    // For each query, rank all texts by similarity
    for (let q = 0; q < queries.length; q++) {
      const query = queries[q]!;
      const queryVec = queryVectors[q]!;

      const scores = texts.map((text, i) => ({
        text: text.substring(0, 70),
        score: cosineSimilarity(queryVec, textVectors[i]!),
      }));

      // Sort by score descending
      scores.sort((a, b) => b.score - a.score);

      console.log(`\n  Query: "${query}"`);
      for (const { text, score } of scores.slice(0, 3)) {
        const bar = "█".repeat(Math.round(score * 30));
        console.log(`    ${score.toFixed(4)} ${bar} ${text}...`);
      }
    }
  }

  console.log("\nDone. Compare the rankings above to see how models differ.");
}

main().catch(console.error);
