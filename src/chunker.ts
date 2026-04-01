import type { Chunk, ChunkOptions } from "./types.js";
import { splitText } from "./chunk-text.js";

function buildChunks(rawChunks: string[]): Chunk[] {
  const chunks: Chunk[] = [];
  let prev: string | undefined;
  for (const raw of rawChunks) {
    const text = raw.trim();
    if (text === prev) continue;
    prev = text;
    if (text.length === 0) continue;
    chunks.push({ text, index: chunks.length, totalChunks: 0 });
  }
  for (const c of chunks) c.totalChunks = chunks.length;
  return chunks;
}

export function chunkText(text: string, options: ChunkOptions = {}): Chunk[] {
  const { maxChunkSize = 1500, overlapSize = 200, format = "text" } = options;

  if (format !== "text") {
    throw new Error(
      `Format "${format}" requires chunkTextAsync(). Use chunkTextAsync() for structured formats.`
    );
  }

  const trimmed = text.trim();
  if (trimmed.length === 0) return [];
  if (trimmed.length <= maxChunkSize) {
    return [{ text: trimmed, index: 0, totalChunks: 1 }];
  }

  const rawChunks = splitText(trimmed, maxChunkSize, overlapSize);
  return buildChunks(rawChunks);
}

export async function chunkTextAsync(
  text: string,
  options: ChunkOptions = {}
): Promise<Chunk[]> {
  const { maxChunkSize = 1500, overlapSize = 200, format = "text" } = options;

  const trimmed = text.trim();
  if (trimmed.length === 0) return [];

  if (format !== "text") {
    const { chunkStructured } = await import("./chunk-structured.js");
    const rawChunks = await chunkStructured(trimmed, options);
    return buildChunks(rawChunks);
  }

  if (trimmed.length <= maxChunkSize) {
    return [{ text: trimmed, index: 0, totalChunks: 1 }];
  }

  return buildChunks(splitText(trimmed, maxChunkSize, overlapSize));
}
