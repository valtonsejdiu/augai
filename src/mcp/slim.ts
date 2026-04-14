import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { getClient, getDenseEmbedder, getMemory } from "./shared.js";
import { SkillIndex } from "./skill-index.js";
import { DISPATCH } from "./tools.js";

const SESSION_ID = crypto.randomUUID();

// Boot sequence: embedder → skill index → memory → stdio loop
const embedder = await getDenseEmbedder();
const skillIndex = new SkillIndex(getClient(), embedder);
await skillIndex.build();
await getMemory(); // warms the session_memory collection

const server = new Server(
  { name: "augai-slim", version: "1.0.0" },
  {
    capabilities: { tools: {} },
    instructions: `You have access to augai — a semantic session memory system backed by Qdrant.

RULE: Call get_skill BEFORE every response. Pass what you are about to do as context.
Execute the skill at priority 1 (required) using execute_skill before proceeding.
The response includes params_schema — use it to construct the correct params.

This ensures relevant memories are always retrieved before you answer.`,
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "get_skill",
      description:
        "ALWAYS call this tool before taking any action or answering any question. " +
        "Provide what you are about to do as `context`. Returns a priority-ranked list " +
        "of 1–3 skills. The skill at priority 1 is REQUIRED — call execute_skill with " +
        "it before proceeding. Each skill includes the exact params_schema to construct.",
      inputSchema: {
        type: "object" as const,
        properties: {
          context: {
            type: "string",
            description: "What you are about to do, in plain language",
          },
        },
        required: ["context"],
      },
    },
    {
      name: "execute_skill",
      description:
        "Execute a skill returned by get_skill. Use the skill name and params_schema " +
        "from the get_skill response to construct params. Always call get_skill first.",
      inputSchema: {
        type: "object" as const,
        properties: {
          skill: {
            type: "string",
            description: "Exact skill name from get_skill response",
          },
          params: {
            type: "object",
            description: "Constructed from the params_schema returned by get_skill",
          },
        },
        required: ["skill"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;

  try {
    if (name === "get_skill") {
      const context = String((args as Record<string, unknown>).context ?? "");
      const skills = await skillIndex.search(context);
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({ skills, session_id: SESSION_ID }),
          },
        ],
      };
    }

    if (name === "execute_skill") {
      // session_id from model input is intentionally ignored — server owns session identity
      const { skill, params = {} } = args as {
        skill: string;
        params?: Record<string, unknown>;
      };

      const handler = DISPATCH[skill];
      if (!handler) {
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: `Unknown skill: ${skill}` }) }],
          isError: true,
        };
      }

      const memory = await getMemory();
      const result = await handler({ session_id: SESSION_ID, ...(params as object) }, memory);
      return {
        content: [{ type: "text" as const, text: JSON.stringify(result) }],
      };
    }

    return {
      content: [{ type: "text" as const, text: JSON.stringify({ error: `Unknown tool: ${name}` }) }],
      isError: true,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
