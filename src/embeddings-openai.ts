import type { DenseEmbedder } from "./types.js";

export type OpenAIModel = "text-embedding-3-small" | "text-embedding-3-large";

const MODELS: Record<OpenAIModel, { dimension: number; maxTokens: number }> = {
  "text-embedding-3-small": { dimension: 1536, maxTokens: 8192 },
  "text-embedding-3-large": { dimension: 3072, maxTokens: 8192 },
};

export class OpenAIEmbedder implements DenseEmbedder {
  private client: import("openai").default | null = null;
  readonly dimension: number;
  readonly maxTokens: number;
  private readonly model: OpenAIModel;
  private readonly customDimensions: number | undefined;

  constructor(model: OpenAIModel = "text-embedding-3-small", dimensions?: number) {
    this.model = model;
    this.customDimensions = dimensions;
    this.dimension = dimensions ?? MODELS[model].dimension;
    this.maxTokens = MODELS[model].maxTokens;
  }

  private async getClient() {
    if (!this.client) {
      if (!process.env.OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY environment variable is required for OpenAI embeddings");
      }
      const { default: OpenAI } = await import("openai");
      this.client = new OpenAI();
    }
    return this.client;
  }

  async embedQuery(text: string): Promise<number[]> {
    const client = await this.getClient();
    const res = await client.embeddings.create({
      model: this.model,
      input: text,
      encoding_format: "float",
      ...(this.customDimensions && { dimensions: this.customDimensions }),
    });
    return res.data[0]!.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const client = await this.getClient();
    const results: number[][] = [];

    for (let i = 0; i < texts.length; i += 2048) {
      const batch = texts.slice(i, i + 2048);
      const res = await client.embeddings.create({
        model: this.model,
        input: batch,
        encoding_format: "float",
        ...(this.customDimensions && { dimensions: this.customDimensions }),
      });
      for (const item of res.data) {
        results.push(item.embedding);
      }
    }
    return results;
  }
}
