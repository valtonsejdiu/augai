import { FastEmbedEmbedder, EmbeddingModel } from "./embeddings.js";
import {
  SidecarDenseEmbedder,
  SidecarSparseEmbedder,
  SidecarReranker,
  checkSidecarHealth,
} from "./sidecar-client.js";

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

const queries = [
  "database storage format",
  "frontend state",
  "container orchestration",
  "deep learning architecture",
];

async function runDenseLocal() {
  console.log("\n--- Pipeline 1: Dense-only (local fastembed, AllMiniLML6V2 384d) ---");
  const embedder = new FastEmbedEmbedder(EmbeddingModel.AllMiniLML6V2);
  const textVecs = await embedder.embedBatch(texts);
  const queryVecs = await embedder.embedBatch(queries);

  for (let q = 0; q < queries.length; q++) {
    const scores = texts.map((text, i) => ({
      text: text.substring(0, 70),
      score: cosineSimilarity(queryVecs[q]!, textVecs[i]!),
    }));
    scores.sort((a, b) => b.score - a.score);
    console.log(`\n  Query: "${queries[q]}"`);
    for (const { text, score } of scores.slice(0, 3)) {
      console.log(`    ${score.toFixed(4)} ${text}...`);
    }
  }
}

async function runDenseSidecar() {
  console.log("\n--- Pipeline 2: Dense-only (sidecar, NomicEmbedTextV15 768d) ---");
  const embedder = new SidecarDenseEmbedder();
  const textVecs = await embedder.embedBatch(texts);
  const queryVecs = await embedder.embedBatch(queries);

  for (let q = 0; q < queries.length; q++) {
    const scores = texts.map((text, i) => ({
      text: text.substring(0, 70),
      score: cosineSimilarity(queryVecs[q]!, textVecs[i]!),
    }));
    scores.sort((a, b) => b.score - a.score);
    console.log(`\n  Query: "${queries[q]}"`);
    for (const { text, score } of scores.slice(0, 3)) {
      console.log(`    ${score.toFixed(4)} ${text}...`);
    }
  }
}

async function runHybrid() {
  console.log("\n--- Pipeline 3: Hybrid (dense + sparse, sidecar) ---");
  const dense = new SidecarDenseEmbedder();
  const sparse = new SidecarSparseEmbedder();

  const textDenseVecs = await dense.embedBatch(texts);
  const textSparseVecs = await sparse.embedBatch(texts);
  const queryDenseVecs = await dense.embedBatch(queries);
  const querySparseVecs = await sparse.embedBatch(queries);

  for (let q = 0; q < queries.length; q++) {
    const denseScores = texts.map((_, i) =>
      cosineSimilarity(queryDenseVecs[q]!, textDenseVecs[i]!)
    );

    // Sparse dot product
    const sparseScores = texts.map((_, i) => {
      const qSparse = querySparseVecs[q]!;
      const tSparse = textSparseVecs[i]!;
      let dot = 0;
      const tMap = new Map<number, number>();
      for (let j = 0; j < tSparse.indices.length; j++) {
        tMap.set(tSparse.indices[j]!, tSparse.values[j]!);
      }
      for (let j = 0; j < qSparse.indices.length; j++) {
        const val = tMap.get(qSparse.indices[j]!);
        if (val !== undefined) dot += qSparse.values[j]! * val;
      }
      return dot;
    });

    // Normalize sparse scores to [0,1]
    const maxSparse = Math.max(...sparseScores, 1e-9);
    const normSparse = sparseScores.map((s) => s / maxSparse);

    // RRF-style fusion
    const scores = texts.map((text, i) => ({
      text: text.substring(0, 70),
      dense: denseScores[i]!,
      sparse: normSparse[i]!,
      score: denseScores[i]! * 0.5 + normSparse[i]! * 0.5,
    }));
    scores.sort((a, b) => b.score - a.score);

    console.log(`\n  Query: "${queries[q]}"`);
    for (const { text, score, dense: d, sparse: s } of scores.slice(0, 3)) {
      console.log(`    ${score.toFixed(4)} (d=${d.toFixed(4)} s=${s.toFixed(4)}) ${text}...`);
    }
  }
}

async function runHybridRerank() {
  console.log("\n--- Pipeline 4: Hybrid + Reranker (full pipeline, sidecar) ---");
  const dense = new SidecarDenseEmbedder();
  const sparse = new SidecarSparseEmbedder();
  const reranker = new SidecarReranker();

  const textDenseVecs = await dense.embedBatch(texts);
  const textSparseVecs = await sparse.embedBatch(texts);
  const queryDenseVecs = await dense.embedBatch(queries);
  const querySparseVecs = await sparse.embedBatch(queries);

  for (let q = 0; q < queries.length; q++) {
    const denseScores = texts.map((_, i) =>
      cosineSimilarity(queryDenseVecs[q]!, textDenseVecs[i]!)
    );

    const sparseScores = texts.map((_, i) => {
      const qSparse = querySparseVecs[q]!;
      const tSparse = textSparseVecs[i]!;
      let dot = 0;
      const tMap = new Map<number, number>();
      for (let j = 0; j < tSparse.indices.length; j++) {
        tMap.set(tSparse.indices[j]!, tSparse.values[j]!);
      }
      for (let j = 0; j < qSparse.indices.length; j++) {
        const val = tMap.get(qSparse.indices[j]!);
        if (val !== undefined) dot += qSparse.values[j]! * val;
      }
      return dot;
    });

    const maxSparse = Math.max(...sparseScores, 1e-9);
    const normSparse = sparseScores.map((s) => s / maxSparse);

    const fusionScores = texts.map((_, i) => denseScores[i]! * 0.5 + normSparse[i]! * 0.5);

    // Rerank top candidates
    const rerankResults = await reranker.rerank(queries[q]!, texts, 3);

    const scores = texts.map((text, i) => {
      const rr = rerankResults.find((r) => r.index === i);
      const rrScore = rr?.relevanceScore ?? 0;
      return {
        text: text.substring(0, 70),
        fusion: fusionScores[i]!,
        rerank: rrScore,
        score: fusionScores[i]! * 0.5 + rrScore * 0.5,
      };
    });
    scores.sort((a, b) => b.score - a.score);

    console.log(`\n  Query: "${queries[q]}"`);
    for (const { text, score, fusion, rerank: rr } of scores.slice(0, 3)) {
      console.log(`    ${score.toFixed(4)} (fuse=${fusion.toFixed(4)} rr=${rr.toFixed(4)}) ${text}...`);
    }
  }
}

async function main() {
  console.log("=".repeat(60));
  console.log("Hybrid Search Pipeline Comparison");
  console.log("=".repeat(60));

  // Pipeline 1 always runs (local, no sidecar needed)
  await runDenseLocal();

  const sidecarUp = await checkSidecarHealth();
  if (!sidecarUp) {
    console.log("\n[!] Sidecar not running — skipping pipelines 2-4");
    console.log("    Start with: cd augai-embed && cargo run --release");
    return;
  }

  await runDenseSidecar();
  await runHybrid();
  await runHybridRerank();

  console.log("\n" + "=".repeat(60));
  console.log("Done. Compare rankings across pipelines above.");
}

main().catch(console.error);
