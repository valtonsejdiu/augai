import type { DenseEmbedder } from "./types.js";

export type MistralModel = "mistral-embed" | "codestral-embed";

export const MODELS: Record<MistralModel, { dimension: number; maxTokens: number }> = {
  "mistral-embed": { dimension: 1024, maxTokens: 8192 },
  "codestral-embed": { dimension: 1024, maxTokens: 8192 },
};

const MAX_BATCH = 512;

export class MistralEmbedder implements DenseEmbedder {
  private client: import("openai").default | null = null;
  readonly dimension: number;
  readonly maxTokens: number;
  private readonly model: MistralModel;
  private readonly customDimensions: number | undefined;
  private readonly minRequestInterval: number;
  private lastRequestTime = 0;

  constructor(
    model: MistralModel = "codestral-embed",
    dimensions?: number,
    requestsPerSecond = 1,
  ) {
    this.model = model;
    this.customDimensions = dimensions;
    this.dimension = dimensions ?? MODELS[model].dimension;
    this.maxTokens = MODELS[model].maxTokens;
    this.minRequestInterval = 1000 / requestsPerSecond;
  }

  private async getClient() {
    if (!this.client) {
      if (!process.env.MISTRAL_API_KEY) {
        throw new Error("MISTRAL_API_KEY environment variable is required for Mistral embeddings");
      }
      const { default: OpenAI } = await import("openai");
      this.client = new OpenAI({
        apiKey: process.env.MISTRAL_API_KEY,
        baseURL: "https://api.mistral.ai/v1",
      });
    }
    return this.client;
  }

  private async throttle() {
    const now = Date.now();
    const elapsed = now - this.lastRequestTime;
    if (elapsed < this.minRequestInterval) {
      await new Promise((r) => setTimeout(r, this.minRequestInterval - elapsed));
    }
    this.lastRequestTime = Date.now();
  }

  async embedQuery(text: string): Promise<number[]> {
    const client = await this.getClient();
    await this.throttle();
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

    for (let i = 0; i < texts.length; i += MAX_BATCH) {
      const batch = texts.slice(i, i + MAX_BATCH);
      await this.throttle();
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
