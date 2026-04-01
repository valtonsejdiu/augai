import type { DenseEmbedder } from "./types.js";

export type OllamaModel =
  | "nomic-embed-text" | "mxbai-embed-large" | "all-minilm"
  | "snowflake-arctic-embed" | "bge-m3" | "nub235/voyage-4-nano"
  | "qwen3-embedding" | "qwen3-embedding:0.6b" | "qwen3-embedding:4b" | "qwen3-embedding:8b";

const MODELS: Record<OllamaModel, { dimension: number; maxTokens: number }> = {
  "nomic-embed-text": { dimension: 768, maxTokens: 8192 },
  "mxbai-embed-large": { dimension: 1024, maxTokens: 512 },
  "all-minilm": { dimension: 384, maxTokens: 256 },
  "snowflake-arctic-embed": { dimension: 1024, maxTokens: 512 },
  "bge-m3": { dimension: 1024, maxTokens: 8192 },
  "nub235/voyage-4-nano": { dimension: 1024, maxTokens: 4096 }, // GGUF port — 1024d (official is 2048d, projection layer lost in conversion)
  "qwen3-embedding": { dimension: 4096, maxTokens: 32000 },
  "qwen3-embedding:0.6b": { dimension: 1024, maxTokens: 32000 },
  "qwen3-embedding:4b": { dimension: 2560, maxTokens: 32000 },
  "qwen3-embedding:8b": { dimension: 4096, maxTokens: 32000 },
};

interface OllamaOptions {
  baseUrl?: string;
}

export class OllamaEmbedder implements DenseEmbedder {
  readonly dimension: number;
  readonly maxTokens: number;
  private readonly model: string;
  private readonly baseUrl: string;

  private constructor(model: string, dimension: number, maxTokens: number, baseUrl: string) {
    this.model = model;
    this.dimension = dimension;
    this.maxTokens = maxTokens;
    this.baseUrl = baseUrl;
  }

  static async create(model?: string, options?: OllamaOptions): Promise<OllamaEmbedder> {
    const name = model ?? "nomic-embed-text";
    const baseUrl = (options?.baseUrl ?? "http://localhost:11434").replace(/\/$/, "");

    if (name in MODELS) {
      const { dimension, maxTokens } = MODELS[name as OllamaModel];
      return new OllamaEmbedder(name, dimension, maxTokens, baseUrl);
    }

    // Unknown model — probe for dimension
    const embedder = new OllamaEmbedder(name, 0, 512, baseUrl);
    const [vec] = await embedder.embed(["dimension probe"]);
    return new OllamaEmbedder(name, vec.length, 512, baseUrl);
  }

  private async embed(input: string[]): Promise<number[][]> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: this.model, input }),
      });
    } catch (err: unknown) {
      // TypeError is Bun/browser fetch's connection error type; ECONNREFUSED covers Node-style cause codes
      if (err instanceof TypeError || (err as any)?.cause?.code === "ECONNREFUSED") {
        throw new Error(`Cannot connect to Ollama at ${this.baseUrl}. Is it running? Start with: ollama serve`);
      }
      throw err;
    }

    if (res.status === 404) {
      throw new Error(`Model "${this.model}" not found in Ollama. Pull it with: ollama pull ${this.model}`);
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Ollama API error (${res.status}): ${body}`);
    }

    const data: unknown = await res.json();
    const embeddings = (data as Record<string, unknown>)?.embeddings;
    if (!Array.isArray(embeddings) || embeddings.length === 0) {
      throw new Error("Unexpected Ollama response: no embeddings returned");
    }
    return embeddings as number[][];
  }

  async embedQuery(text: string): Promise<number[]> {
    const [vec] = await this.embed([text]);
    return vec;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const results: number[][] = [];

    for (let i = 0; i < texts.length; i += 128) {
      const batch = texts.slice(i, i + 128);
      const vecs = await this.embed(batch);
      results.push(...vecs);
    }
    return results;
  }
}
