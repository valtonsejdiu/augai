import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { getMemory } from "./shared.js";
import { SKILLS } from "./skill-index.js";
import { DISPATCH } from "./tools.js";

const SESSION_ID = crypto.randomUUID();

const server = new Server(
  { name: "augai-full", version: "1.0.0" },
  {
    capabilities: { tools: {} },
    instructions: `You have access to augai session memory backed by Qdrant.
Before answering any question about past context, documents, or decisions:
call memory_search first. Do not guess from context — search first.

Your session ID for this conversation is: ${SESSION_ID}
Pass this exact value as session_id to every tool call.`,
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: SKILLS.map((skill) => ({
    name: skill.name,
    description: skill.description,
    inputSchema: skill.inputSchema,
  })),
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;

  const handler = DISPATCH[name];
  if (!handler) {
    return {
      content: [{ type: "text" as const, text: JSON.stringify({ error: `Unknown tool: ${name}` }) }],
      isError: true,
    };
  }

  try {
    const memory = await getMemory();
    const result = await handler({ ...(args as object), session_id: SESSION_ID }, memory);
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result) }],
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
