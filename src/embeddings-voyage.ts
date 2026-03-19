import type { DenseEmbedder } from "./types.js";

export type VoyageModel =
  | "voyage-4-large" | "voyage-4" | "voyage-4-lite"
  | "voyage-3-large" | "voyage-3" | "voyage-3-lite";

export const VOYAGE_MODELS: VoyageModel[] = [
  "voyage-4-large", "voyage-4", "voyage-4-lite",
  "voyage-3-large", "voyage-3", "voyage-3-lite",
];

export interface VoyageStrategy {
  indexModel: VoyageModel;
  queryModel: VoyageModel;
  fallbackModel: VoyageModel;
}

const MODELS: Record<VoyageModel, { dimension: number; maxTokens: number }> = {
  "voyage-4-large": { dimension: 1024, maxTokens: 16000 },
  "voyage-4":       { dimension: 1024, maxTokens: 16000 },
  "voyage-4-lite":  { dimension: 1024, maxTokens: 16000 },
  "voyage-3-large": { dimension: 1024, maxTokens: 4000 },
  "voyage-3":       { dimension: 1024, maxTokens: 4000 },
  "voyage-3-lite":  { dimension: 512,  maxTokens: 4000 },
};

export class VoyageEmbedder implements DenseEmbedder {
  private client: import("voyageai").VoyageAIClient | null = null;
  readonly dimension: number;
  readonly maxTokens: number;
  private readonly indexModel: VoyageModel;
  private readonly queryModel: VoyageModel;
  private readonly fallbackModel: VoyageModel;
  private readonly outputDimension: number | undefined;

  constructor(modelOrStrategy?: VoyageModel | VoyageStrategy, outputDimension?: number) {
    if (modelOrStrategy && typeof modelOrStrategy === "object") {
      this.indexModel = modelOrStrategy.indexModel;
      this.queryModel = modelOrStrategy.queryModel;
      this.fallbackModel = modelOrStrategy.fallbackModel;

      const idxDim = MODELS[this.indexModel].dimension;
      const qDim = MODELS[this.queryModel].dimension;
      const fbDim = MODELS[this.fallbackModel].dimension;
      if (qDim !== idxDim || fbDim !== idxDim) {
        throw new Error(
          `Voyage model dimension mismatch: indexModel=${this.indexModel}(${idxDim}d) ` +
          `queryModel=${this.queryModel}(${qDim}d) fallbackModel=${this.fallbackModel}(${fbDim}d). ` +
          `All three must share the same output dimension.`
        );
      }
    } else {
      const m = (modelOrStrategy as VoyageModel | undefined) ?? "voyage-4-large";
      this.indexModel = m;
      this.queryModel = m;
      this.fallbackModel = m;
    }

    this.outputDimension = outputDimension;
    this.dimension = outputDimension ?? MODELS[this.indexModel].dimension;
    this.maxTokens = MODELS[this.indexModel].maxTokens;
  }

  private async getClient() {
    if (!this.client) {
      if (!process.env.VOYAGEAI_API_KEY) {
        throw new Error("VOYAGEAI_API_KEY environment variable is required for Voyage AI embeddings");
      }
      const { VoyageAIClient } = await import("voyageai");
      this.client = new VoyageAIClient({ apiKey: process.env.VOYAGEAI_API_KEY });
    }
    return this.client;
  }

  private isRetryable(err: unknown): boolean {
    if (err && typeof err === "object" && "statusCode" in err) {
      const code = (err as { statusCode: number }).statusCode;
      return code === 429 || code >= 500;
    }
    return false;
  }

  async embedQuery(text: string): Promise<number[]> {
    const client = await this.getClient();
    const call = (model: VoyageModel) =>
      client.embed({
        model,
        input: [text],
        inputType: "query",
        ...(this.outputDimension && { outputDimension: this.outputDimension }),
      });

    try {
      const res = await call(this.queryModel);
      return res.data![0]!.embedding!;
    } catch (err) {
      if (!this.isRetryable(err) || this.fallbackModel === this.queryModel) throw err;
      console.warn(`[voyage] embedQuery: falling back from ${this.queryModel} → ${this.fallbackModel}`);
      const res = await call(this.fallbackModel);
      return res.data![0]!.embedding!;
    }
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const client = await this.getClient();
    const results: number[][] = [];

    const callBatch = (model: VoyageModel, batch: string[]) =>
      client.embed({
        model,
        input: batch,
        inputType: "document",
        ...(this.outputDimension && { outputDimension: this.outputDimension }),
      });

    for (let i = 0; i < texts.length; i += 128) {
      const batch = texts.slice(i, i + 128);
      let res;
      try {
        res = await callBatch(this.indexModel, batch);
      } catch (err) {
        if (!this.isRetryable(err) || this.fallbackModel === this.indexModel) throw err;
        console.warn(`[voyage] embedBatch: falling back from ${this.indexModel} → ${this.fallbackModel}`);
        res = await callBatch(this.fallbackModel, batch);
      }
      for (const item of res.data!) {
        results.push(item.embedding!);
      }
    }
    return results;
  }
}
