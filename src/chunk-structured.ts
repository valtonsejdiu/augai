import type { ChunkOptions } from "./types.js";

const SUPPORTED_LANGUAGES = new Set([
  "cpp", "go", "java", "js", "php", "proto", "python", "rst", "ruby",
  "rust", "scala", "swift", "markdown", "latex", "html", "sol",
]);

export async function chunkStructured(
  text: string,
  options: ChunkOptions
): Promise<string[]> {
  process.env.LANGCHAIN_TRACING_V2 ??= "false";

  const { maxChunkSize = 1500, overlapSize = 200, format, language } = options;

  let mod;
  try {
    mod = await import("@langchain/textsplitters");
  } catch {
    throw new Error(
      'Install @langchain/textsplitters for structured format support: bun add @langchain/textsplitters @langchain/core'
    );
  }

  const { RecursiveCharacterTextSplitter, MarkdownTextSplitter, LatexTextSplitter } = mod;
  const config = { chunkSize: maxChunkSize, chunkOverlap: overlapSize };

  let splitter;
  switch (format) {
    case "markdown":
      splitter = new MarkdownTextSplitter(config);
      break;
    case "html":
      splitter = RecursiveCharacterTextSplitter.fromLanguage("html", config);
      break;
    case "latex":
      splitter = new LatexTextSplitter(config);
      break;
    case "code": {
      const lang = language ?? "js";
      if (!SUPPORTED_LANGUAGES.has(lang)) {
        throw new Error(
          `Unsupported code language: "${lang}". Supported: ${[...SUPPORTED_LANGUAGES].join(", ")}`
        );
      }
      splitter = RecursiveCharacterTextSplitter.fromLanguage(lang as any, config);
      break;
    }
    default:
      throw new Error(`Unsupported structured format: ${format}`);
  }

  return splitter.splitText(text);
}
