import type { DenseEmbedder } from "./types.js";

export type ProviderName = "local" | "openai" | "cohere" | "voyage" | "ollama" | "rust" | "mistral";

export async function createEmbedder(provider?: ProviderName): Promise<DenseEmbedder> {
  const name = provider ?? (process.env.AUGAI_EMBEDDER as ProviderName | undefined) ?? "ollama";

  switch (name) {
    case "local": {
      const { FastEmbedEmbedder } = await import("./embeddings.js");
      return new FastEmbedEmbedder();
    }
    case "openai": {
      const { OpenAIEmbedder } = await import("./embeddings-openai.js");
      return new OpenAIEmbedder();
    }
    case "cohere": {
      const { CohereEmbedder } = await import("./embeddings-cohere.js");
      return new CohereEmbedder();
    }
    case "voyage": {
      const { VoyageEmbedder, VOYAGE_MODELS } = await import("./embeddings-voyage.js");
      const resolve = (env: string | undefined, def: string) => {
        const v = (env ?? def) as (typeof VOYAGE_MODELS)[number];
        if (!VOYAGE_MODELS.includes(v)) throw new Error(`Invalid Voyage model "${v}". Valid: ${VOYAGE_MODELS.join(", ")}`);
        return v;
      };
      return new VoyageEmbedder({
        indexModel: resolve(process.env.VOYAGE_INDEX_MODEL, "voyage-4-large"),
        queryModel: resolve(process.env.VOYAGE_QUERY_MODEL, "voyage-4-lite"),
        fallbackModel: resolve(process.env.VOYAGE_FALLBACK_MODEL, "voyage-4"),
      });
    }
    case "ollama": {
      const { OllamaEmbedder } = await import("./embeddings-ollama.js");
      return OllamaEmbedder.create(process.env.OLLAMA_MODEL, {
        baseUrl: process.env.OLLAMA_HOST,
      });
    }
    case "mistral": {
      const { MistralEmbedder, MODELS: MISTRAL_MODELS } = await import("./embeddings-mistral.js");
      const model = (process.env.MISTRAL_MODEL ?? "codestral-embed") as keyof typeof MISTRAL_MODELS;
      if (!(model in MISTRAL_MODELS)) throw new Error(`Invalid Mistral model "${model}". Valid: ${Object.keys(MISTRAL_MODELS).join(", ")}`);
      const dim = process.env.MISTRAL_DIMENSIONS ? Number(process.env.MISTRAL_DIMENSIONS) : undefined;
      const rps = process.env.MISTRAL_RPS ? Number(process.env.MISTRAL_RPS) : 1;
      return new MistralEmbedder(model, dim, rps);
    }
    case "rust": {
      const { SidecarDenseEmbedder } = await import("./sidecar-client.js");
      return new SidecarDenseEmbedder();
    }
    default:
      throw new Error(`Unknown embedding provider: "${name}". Use: local, openai, cohere, voyage, ollama, rust, mistral`);
  }
}
