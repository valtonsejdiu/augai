// Text chunking for embedding models.
//
// Embedding models have a token limit (~512 tokens ≈ ~2000 chars).
// Long text gets silently truncated, losing information.
// Chunking splits text into overlapping pieces so each fits the model's
// context window while preserving boundary context via overlap.

export interface ChunkOptions {
  maxChunkSize?: number;  // Max characters per chunk (default: 1500 ≈ 375 tokens)
  overlapSize?: number;   // Characters of overlap between chunks (default: 200)
}

export interface Chunk {
  text: string;
  index: number;       // Position within the source (0-based)
  totalChunks: number; // How many chunks the source was split into
}

// Split text into sentence-boundary-aligned chunks with overlap.
//
// Algorithm:
// 1. Split on sentence boundaries (.!? followed by whitespace or end)
// 2. Accumulate sentences until hitting maxChunkSize
// 3. Start next chunk by rewinding overlapSize chars worth of trailing sentences
// 4. Single sentences exceeding maxChunkSize get hard-split by character
export function chunkText(
  text: string,
  options: ChunkOptions = {}
): Chunk[] {
  const { maxChunkSize = 1500, overlapSize = 200 } = options;

  // Short text — no chunking needed
  if (text.length <= maxChunkSize) {
    return [{ text, index: 0, totalChunks: 1 }];
  }

  // Split into sentences. Keeps the delimiter attached to the sentence.
  const matched = text.match(/[^.!?]*[.!?]+[\s]*/g);
  const sentences: string[] = matched ? [...matched] : [];

  // If regex found nothing (no sentence-ending punctuation), treat whole text as one "sentence"
  if (sentences.length === 0) {
    sentences.push(text);
  }

  // Check if there's trailing text after the last sentence boundary
  const joined = sentences.join("");
  if (joined.length < text.length) {
    sentences.push(text.slice(joined.length));
  }

  const rawChunks: string[] = [];
  let i = 0;

  while (i < sentences.length) {
    let chunk = "";
    const startIdx = i;

    // Accumulate sentences until we hit the size limit
    while (i < sentences.length && (chunk + sentences[i]!).length <= maxChunkSize) {
      chunk += sentences[i]!;
      i++;
    }

    // If we couldn't fit even one sentence, hard-split it
    if (i === startIdx) {
      const longSentence = sentences[i]!;
      for (let pos = 0; pos < longSentence.length; pos += maxChunkSize) {
        rawChunks.push(longSentence.slice(pos, pos + maxChunkSize));
      }
      i++;
      continue;
    }

    rawChunks.push(chunk);

    // Rewind for overlap: back up enough sentences to cover overlapSize chars
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

  // Deduplicate consecutive identical chunks (can happen with overlap logic)
  const deduped = rawChunks.filter((c, idx) => idx === 0 || c !== rawChunks[idx - 1]);

  const totalChunks = deduped.length;
  return deduped.map((text, index) => ({
    text: text.trim(),
    index,
    totalChunks,
  }));
}
