import type { Reranker } from "./types.js";

export type RerankerProviderName = "infinity" | "cohere" | "voyage" | "local" | "rust";

export async function createReranker(
  provider?: RerankerProviderName
): Promise<Reranker | null> {
  const name = provider ?? (process.env.AUGAI_RERANKER as RerankerProviderName | undefined);
  if (!name) return null;

  switch (name) {
    case "infinity": {
      const { InfinityReranker } = await import("./rerank-infinity.js");
      return new InfinityReranker();
    }
    case "cohere": {
      const { CohereReranker } = await import("./reranker-cohere.js");
      return new CohereReranker();
    }
    case "voyage": {
      const { VoyageReranker } = await import("./reranker-voyage.js");
      return new VoyageReranker();
    }
    case "local": {
      const { LocalReranker } = await import("./reranker-local.js");
      return new LocalReranker();
    }
    case "rust": {
      const { SidecarReranker } = await import("./sidecar-client.js");
      return new SidecarReranker();
    }
    default:
      throw new Error(`Unknown reranker provider: "${name}". Use: infinity, cohere, voyage, local, rust`);
  }
}
