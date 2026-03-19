import type { Chunk, ChunkOptions } from "./types.js";

export function chunkText(
  text: string,
  options: ChunkOptions = {}
): Chunk[] {
  const { maxChunkSize = 1500, overlapSize = 200 } = options;

  if (text.length <= maxChunkSize) {
    return [{ text, index: 0, totalChunks: 1 }];
  }

  const matched = text.match(/[^.!?]*[.!?]+[\s]*/g);
  const sentences: string[] = matched ? [...matched] : [];

  if (sentences.length === 0) {
    sentences.push(text);
  }

  const matchedLen = sentences.reduce((sum, s) => sum + s.length, 0);
  if (matchedLen < text.length) {
    sentences.push(text.slice(matchedLen));
  }

  const rawChunks: string[] = [];
  let i = 0;

  while (i < sentences.length) {
    let chunk = "";
    const startIdx = i;

    while (i < sentences.length && (chunk + sentences[i]!).length <= maxChunkSize) {
      chunk += sentences[i]!;
      i++;
    }

    if (i === startIdx) {
      const longSentence = sentences[i]!;
      for (let pos = 0; pos < longSentence.length; pos += maxChunkSize) {
        rawChunks.push(longSentence.slice(pos, pos + maxChunkSize));
      }
      i++;
      continue;
    }

    rawChunks.push(chunk);

    if (i < sentences.length) {
      let overlapLen = 0;
      let rewind = 0;
      for (let j = i - 1; j >= startIdx; j--) {
        overlapLen += sentences[j]!.length;
        rewind++;
        if (overlapLen >= overlapSize) break;
      }
      i -= rewind;
    }
  }

  // overlap can produce identical consecutive chunks
  const deduped = rawChunks.filter((c, idx) => idx === 0 || c !== rawChunks[idx - 1]);

  const totalChunks = deduped.length;
  return deduped.map((text, index) => ({ text: text.trim(), index, totalChunks }));
}
