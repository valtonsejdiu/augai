import type { QdrantClient } from "@qdrant/js-client-rest";
import type { DenseEmbedder } from "../types.js";

export interface SkillDefinition {
  name: string;
  description: string;
  inputSchema: {
    type: "object";
    properties: Record<string, { type: string; description: string; default?: unknown }>;
    required: string[];
  };
  required: boolean;
  reason: string;
}

export interface SkillSuggestion {
  priority: number;
  skill: string;
  required: boolean;
  reason: string;
  input_schema: {
    type: "object";
    properties: Record<string, { type: string; description: string; default?: unknown }>;
    required: string[];
  };
}

export const SKILLS: SkillDefinition[] = [
  {
    name: "memory_search",
    required: true,
    reason: "Retrieve relevant memories before responding — always run first",
    inputSchema: {
      type: "object",
      properties: {
        query:       { type: "string",  description: "What to search for" },
        session_id:  { type: "string",  description: "Limit results to one session" },
        limit:       { type: "number",  description: "Max results to return", default: 5 },
        time_weight: { type: "boolean", description: "Apply recency scoring", default: true },
      },
      required: ["query"],
    },
    description: `Retrieves the most semantically relevant memories from past conversations and ingested documents using hybrid vector search with optional reranking and time-weighted scoring.

WHEN TO USE: before answering any question that might reference past context; when the user asks about something previously discussed; when you need to verify a fact, decision, or person before responding; when a project, document, or event was mentioned before.

SIGNALS: "what did we decide", "do you remember", "we talked about this", "find relevant context", "check the docs", "was this mentioned", any factual question about prior sessions.

NOT FOR: browsing recent messages in order — use memory_list_session for that.
NOT FOR: storing new content — use memory_add or memory_ingest for that.`,
  },

  {
    name: "memory_add",
    required: false,
    reason: "Store this exchange in memory after responding",
    inputSchema: {
      type: "object",
      properties: {
        text:       { type: "string", description: "Text to store verbatim" },
        session_id: { type: "string", description: "Session this belongs to" },
        role:       { type: "string", description: "user | assistant | system" },
      },
      required: ["text", "session_id", "role"],
    },
    description: `Stores a message, fact, or exchange into session memory as a vector point. Content is chunked, embedded, and upserted into Qdrant for future retrieval via memory_search.

WHEN TO USE: after generating a response worth remembering; when the user shares a fact, decision, or preference; when storing the outcome of a session for later recall.

SIGNALS: "remember this", "save that", "store for later", end-of-session archiving, recording a decision or conclusion.

NOT FOR: retrieving memories — use memory_search for that.
NOT FOR: ingesting files — use memory_ingest for that.`,
  },

  {
    name: "memory_ingest",
    required: false,
    reason: "Ingest the referenced file into memory",
    inputSchema: {
      type: "object",
      properties: {
        path:       { type: "string", description: "Absolute or relative file path" },
        session_id: { type: "string", description: "Session to associate with" },
      },
      required: ["path", "session_id"],
    },
    description: `Reads a local file (PDF, Markdown, HTML, plain text), chunks it, embeds each chunk, and stores all chunks in session memory with SHA-256 deduplication. Already-ingested files (same hash) are skipped.

WHEN TO USE: when the user points to a file and wants it searchable; when onboarding documents into a session; when adding reference material for later retrieval.

SIGNALS: "load this file", "ingest", "add this document", "read this PDF", a file path is mentioned, the user wants to make a document searchable.

NOT FOR: storing plain text typed in conversation — use memory_add for that.
NOT FOR: listing what's already ingested — use memory_list_documents for that.`,
  },

  {
    name: "memory_list_documents",
    required: false,
    reason: "List previously ingested documents for this session",
    inputSchema: {
      type: "object",
      properties: {
        session_id: { type: "string", description: "Session to list documents for" },
      },
      required: ["session_id"],
    },
    description: `Returns a list of all files ingested into a session: filename, type, chunk count, and ingest timestamp. Does not return document content — use memory_search for content.

WHEN TO USE: when the user asks what documents are loaded; to verify an ingest succeeded; to audit the knowledge base for a session before answering.

SIGNALS: "what files do you have", "what's ingested", "list documents", "what did I load", "show me what's in memory".

NOT FOR: searching document content — use memory_search for that.
NOT FOR: browsing conversation history — use memory_list_session for that.`,
  },

  {
    name: "memory_list_session",
    required: false,
    reason: "Show recent conversation history in chronological order",
    inputSchema: {
      type: "object",
      properties: {
        session_id: { type: "string", description: "Session to list" },
        limit:      { type: "number", description: "Max messages to return", default: 20 },
      },
      required: ["session_id"],
    },
    description: `Returns recent messages for a session in chronological order (oldest first). Shows the raw conversation timeline — who said what and when. Scores are 0 (this is not a search).

WHEN TO USE: when the user wants to review what was said recently; to reconstruct the flow of a conversation; when order and recency matter more than relevance.

SIGNALS: "show me the history", "what did we say", "scroll back", "recent messages", "what happened in this session".

NOT FOR: semantic retrieval of relevant content — use memory_search for that.
NOT FOR: listing ingested files — use memory_list_documents for that.`,
  },

  {
    name: "memory_clear_session",
    required: false,
    reason: "Clear all memory for this session — irreversible",
    inputSchema: {
      type: "object",
      properties: {
        session_id: { type: "string", description: "Session to clear" },
      },
      required: ["session_id"],
    },
    description: `Deletes all vector points associated with a session from Qdrant. Irreversible. Affects both conversation messages and ingested document chunks stored under this session ID.

WHEN TO USE: when the user explicitly asks to clear or reset memory; at the start of a fresh context where prior session data would be noise; when cleaning up after testing.

SIGNALS: "clear memory", "start fresh", "forget everything", "reset session", "wipe context".

NOT FOR: any retrieval or storage — this is a destructive reset operation only.`,
  },
];

export class SkillIndex {
  private static COLLECTION = "augai_skills";
  private client: QdrantClient;
  private embedder: DenseEmbedder;

  constructor(client: QdrantClient, embedder: DenseEmbedder) {
    this.client = client;
    this.embedder = embedder;
  }

  async build(): Promise<void> {
    const collections = await this.client.getCollections();
    const exists = collections.collections.some((c) => c.name === SkillIndex.COLLECTION);

    if (exists) {
      const info = await this.client.getCollection(SkillIndex.COLLECTION);
      const vectors = info.config?.params?.vectors as Record<string, { size?: number }> | undefined;
      const existingDim = vectors?.dense?.size;
      if (existingDim === this.embedder.dimension) {
        await this._upsert();
        return;
      }
      await this.client.deleteCollection(SkillIndex.COLLECTION);
    }

    await this.client.createCollection(SkillIndex.COLLECTION, {
      vectors: { dense: { size: this.embedder.dimension, distance: "Cosine" } },
    });
    await this._upsert();
  }

  private async _upsert(): Promise<void> {
    const texts = SKILLS.map((s) => s.description);
    const vectors = await this.embedder.embedBatch(texts);

    await this.client.upsert(SkillIndex.COLLECTION, {
      points: SKILLS.map((skill, i) => ({
        id: i + 1,
        vector: { dense: vectors[i]! },
        payload: {
          name:     skill.name,
          required: skill.required,
          reason:   skill.reason,
        },
      })),
    });
  }

  async search(context: string, topK = 3): Promise<SkillSuggestion[]> {
    const queryVec = await this.embedder.embedQuery(context);

    // +1 to account for deduplication of the required skill if it appears in results
    const results = await this.client.search(SkillIndex.COLLECTION, {
      vector: { name: "dense", vector: queryVec },
      limit: topK + 1,
      with_payload: true,
    });

    const seen = new Set<string>();
    const ranked: SkillSuggestion[] = [];

    const requiredSkill = SKILLS.find((s) => s.required)!;
    seen.add(requiredSkill.name);
    ranked.push({
      priority: 1,
      skill: requiredSkill.name,
      required: true,
      reason: requiredSkill.reason,
      input_schema: requiredSkill.inputSchema,
    });

    for (const hit of results) {
      const name = hit.payload!.name as string;
      if (seen.has(name)) continue;
      seen.add(name);
      const skillDef = SKILLS.find((s) => s.name === name)!;
      ranked.push({
        priority: ranked.length + 1,
        skill: name,
        required: false,
        reason: hit.payload!.reason as string,
        input_schema: skillDef.inputSchema,
      });
      if (ranked.length >= topK) break;
    }

    return ranked;
  }
}
