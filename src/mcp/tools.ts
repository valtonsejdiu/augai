import { ingestFile } from "../ingest.js";
import type { SessionMemory } from "../memory.js";

export const DISPATCH: Record<
  string,
  (params: Record<string, unknown>, memory: SessionMemory) => Promise<unknown>
> = {
  memory_search: async (p, m) => {
    if (typeof p.query !== "string" || !p.query)
      throw new Error("memory_search: query is required and must be a non-empty string");
    const results = await m.search(p.query, {
      limit: typeof p.limit === "number" ? p.limit : 5,
      sessionId: typeof p.session_id === "string" ? p.session_id : undefined,
      timeWeight: typeof p.time_weight === "boolean" ? p.time_weight : true,
    });
    return {
      results: results.map((r) => ({
        text: r.text,
        role: r.role,
        score: r.score,
        timestamp: r.timestamp,
      })),
    };
  },

  memory_add: async (p, m) => {
    if (typeof p.text !== "string" || !p.text)
      throw new Error("memory_add: text is required and must be a non-empty string");
    if (typeof p.session_id !== "string" || !p.session_id)
      throw new Error("memory_add: session_id is required");
    if (typeof p.role !== "string" || !p.role)
      throw new Error("memory_add: role is required");
    await m.add(p.text, p.session_id, p.role);
    return { stored: true };
  },

  memory_ingest: async (p, m) => {
    if (typeof p.path !== "string" || !p.path)
      throw new Error("memory_ingest: path is required and must be a non-empty string");
    if (typeof p.session_id !== "string" || !p.session_id)
      throw new Error("memory_ingest: session_id is required");
    const result = await ingestFile(p.path, m, p.session_id);
    return {
      filename: result.filename,
      chunks: result.chunks,
      skipped: result.skipped,
      time_ms: result.timeMs,
    };
  },

  memory_list_documents: async (p, m) => {
    if (typeof p.session_id !== "string" || !p.session_id)
      throw new Error("memory_list_documents: session_id is required");
    const docs = await m.listDocuments(p.session_id);
    return {
      documents: docs.map((d) => ({
        source_file: d.sourceFile,
        source_type: d.sourceType,
        chunks: d.chunks,
        timestamp: d.timestamp,
      })),
    };
  },

  memory_list_session: async (p, m) => {
    if (typeof p.session_id !== "string" || !p.session_id)
      throw new Error("memory_list_session: session_id is required");
    const messages = await m.listSession(
      p.session_id,
      typeof p.limit === "number" ? p.limit : 20,
    );
    return {
      messages: messages.map((msg) => ({
        text: msg.text,
        role: msg.role,
        timestamp: msg.timestamp,
      })),
    };
  },

  memory_clear_session: async (p, m) => {
    if (typeof p.session_id !== "string" || !p.session_id)
      throw new Error("memory_clear_session: session_id is required");
    await m.clearSession(p.session_id);
    return { cleared: true };
  },
};
