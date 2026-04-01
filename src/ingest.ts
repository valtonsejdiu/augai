import { basename, extname } from "node:path";
import type { SessionMemory } from "./memory.js";
import type { ChunkOptions } from "./types.js";

const SUPPORTED_EXTS = new Set([".pdf", ".html", ".htm", ".md", ".txt"]);

export interface IngestResult {
  filename: string;
  chunks: number;
  skipped: boolean;
  timeMs: number;
}

export async function resolveIngestPaths(input: string): Promise<string[]> {
  if (input.includes("*") || input.includes("?")) {
    const glob = new Bun.Glob(input);
    const paths: string[] = [];
    for await (const path of glob.scan(".")) {
      if (SUPPORTED_EXTS.has(extname(path).toLowerCase())) paths.push(path);
    }
    return paths;
  }
  return SUPPORTED_EXTS.has(extname(input).toLowerCase()) ? [input] : [];
}

export async function ingestFile(
  path: string,
  memory: SessionMemory,
  sessionId: string,
  chunkOptions?: ChunkOptions
): Promise<IngestResult> {
  const start = Date.now();
  const filename = basename(path);
  const ext = extname(path).toLowerCase().slice(1);

  const rawBuffer = await Bun.file(path).arrayBuffer();
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(rawBuffer);
  const fileHash = hasher.digest("hex");

  const exists = await memory.findByFileHash(fileHash);
  if (exists) {
    return { filename, chunks: 0, skipped: true, timeMs: Date.now() - start };
  }

  let text: string;
  if (ext === "pdf") {
    let mod;
    try {
      mod = await import("unpdf");
    } catch {
      throw new Error("Install unpdf for PDF support: bun add unpdf");
    }
    const { text: extracted } = await mod.extractText(new Uint8Array(rawBuffer));
    text = Array.isArray(extracted) ? extracted.join("\n") : extracted;
  } else {
    text = new TextDecoder().decode(rawBuffer);
  }

  const sourceType = ext === "htm" ? "html" : ext;
  const chunks = await memory.addDocument(text, sessionId, { sourceFile: filename, sourceType, fileHash }, chunkOptions);

  return { filename, chunks, skipped: false, timeMs: Date.now() - start };
}
