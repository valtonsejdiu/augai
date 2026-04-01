const ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
  "dept", "est", "govt", "inc", "corp", "ltd", "co", "vs", "etc",
  "approx", "appt", "apt", "fig", "ft", "gen", "gov", "no", "vol",
  "rev", "sgt", "capt", "cpl", "maj", "lt", "cmdr", "adm", "pvt",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
  "nov", "dec", "ie", "eg", "cf", "al", "ed", "trans", "tel", "fax",
  "ref", "pp", "pt", "ch", "sec", "ext", "op", "assn", "natl", "intl",
]);

function isSentenceBoundary(text: string, pos: number): boolean {
  const ch = text[pos];
  if (ch !== "." && ch !== "!" && ch !== "?") return false;

  if (pos + 1 < text.length && !/\s/.test(text[pos + 1]!)) return false;

  if (ch === ".") {
    if (pos + 1 < text.length && text[pos + 1] === ".") return false;

    let dotRun = 1;
    for (let j = pos - 1; j >= 0 && text[j] === "."; j--) dotRun++;

    if (dotRun >= 3) return true;
    if (dotRun === 2) return false;

    let ws = pos - 1;
    while (ws >= 0 && /[a-zA-Z.]/.test(text[ws]!)) ws--;
    ws++;

    const rawWord = text.slice(ws, pos);

    if (rawWord.includes(".")) return false;

    const word = rawWord.toLowerCase();
    if (word.length > 0 && word.length <= 6 && ABBREVIATIONS.has(word)) return false;

    if (rawWord.length === 1 && /[A-Z]/.test(rawWord)) return false;
  }

  return true;
}

function splitSentences(text: string): string[] {
  if (text.length === 0) return [];

  const breaks: number[] = [];
  for (let i = 0; i < text.length; i++) {
    if (!isSentenceBoundary(text, i)) continue;
    let end = i + 1;
    while (end < text.length && text[end] === " ") end++;
    if (
      end < text.length &&
      text[end] === "\n" &&
      !(end + 1 < text.length && text[end + 1] === "\n")
    )
      end++;
    breaks.push(end);
  }

  if (breaks.length === 0) return [text];

  const result: string[] = [];
  let start = 0;
  for (const brk of breaks) {
    if (brk > start) result.push(text.slice(start, brk));
    start = brk;
  }
  if (start < text.length) result.push(text.slice(start));

  return result;
}

function splitOnPattern(text: string, pattern: RegExp): string[] {
  const parts = text.split(pattern);
  if (parts.length <= 1) return [text];
  const result: string[] = [];
  for (let i = 0; i < parts.length; i += 2) {
    const content = parts[i] ?? "";
    const sep = parts[i + 1] ?? "";
    const combined = content + sep;
    if (combined.length > 0) result.push(combined);
  }
  return result;
}

type SplitFn = (text: string) => string[];

const SPLIT_HIERARCHY: SplitFn[] = [
  (text) => splitOnPattern(text, /(\n\s*\n)/),
  (text) => splitOnPattern(text, /(\n)/),
  splitSentences,
  (text) => splitOnPattern(text, /( +)/),
];

function splitRecursive(
  text: string,
  maxSize: number,
  level: number = 0
): string[] {
  if (text.length <= maxSize) return [text];

  if (level >= SPLIT_HIERARCHY.length) {
    const pieces: string[] = [];
    for (let i = 0; i < text.length; i += maxSize) {
      pieces.push(text.slice(i, i + maxSize));
    }
    return pieces;
  }

  const segments = SPLIT_HIERARCHY[level]!(text);

  if (segments.length <= 1) return splitRecursive(text, maxSize, level + 1);

  const result: string[] = [];
  for (const seg of segments) {
    if (seg.length <= maxSize) {
      result.push(seg);
    } else {
      result.push(...splitRecursive(seg, maxSize, level + 1));
    }
  }
  return result;
}

function mergeWithOverlap(
  segments: string[],
  maxSize: number,
  overlapSize: number
): string[] {
  const chunks: string[] = [];
  let i = 0;

  while (i < segments.length) {
    let chunk = "";
    const startIdx = i;

    while (i < segments.length && chunk.length + segments[i]!.length <= maxSize) {
      chunk += segments[i]!;
      i++;
    }

    if (i === startIdx) {
      chunk = segments[i]!;
      i++;
    }

    chunks.push(chunk);

    if (i < segments.length && overlapSize > 0) {
      let overlapLen = 0;
      let rewind = 0;
      for (let j = i - 1; j >= startIdx; j--) {
        const segLen = segments[j]!.length;
        if (overlapLen + segLen > overlapSize && rewind > 0) break;
        overlapLen += segLen;
        rewind++;
      }
      if (rewind > 0) i -= rewind;
    }
  }

  return chunks;
}

export function splitText(
  text: string,
  maxSize: number,
  overlapSize: number
): string[] {
  const effectiveOverlap = Math.min(overlapSize, Math.floor(maxSize * 0.5));
  const segments = splitRecursive(text, maxSize);
  return mergeWithOverlap(segments, maxSize, effectiveOverlap);
}
