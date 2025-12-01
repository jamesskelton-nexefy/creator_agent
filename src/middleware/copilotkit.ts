/**
 * Middleware for integrating CopilotKit with deep agents via state emission.
 *
 * This middleware:
 * 1. Extends the agent state with CopilotKit properties and plannedNodes
 * 2. Injects context into the system prompt
 * 3. Tells the agent to output structured JSON for node creation
 *
 * The agent does NOT call frontend tools directly. Instead:
 * - Agent outputs structured data (plannedNodes) in its response
 * - Frontend watches state via useCoAgent and creates nodes directly
 *
 * Based on CopilotKit shared state pattern:
 * https://docs.copilotkit.ai/langgraph/shared-state
 */

import { createMiddleware, type Middleware } from "langchain";
import { z as z3 } from "zod/v3";

/**
 * Schema for CopilotKit context items (from useCopilotReadable)
 */
const CopilotKitContextItemSchema = z3.object({
  description: z3.string(),
  value: z3.any(),
});

/**
 * Schema for a planned node that the agent outputs
 */
const PlannedNodeSchema = z3.object({
  /** Unique identifier for the node */
  id: z3.string(),
  /** Title/name of the node */
  title: z3.string(),
  /** Content or description for the node */
  content: z3.string().optional(),
  /** Parent node ID for hierarchical structures */
  parentId: z3.string().optional(),
  /** Type of node (module, lesson, topic, etc.) */
  type: z3.string().optional(),
  /** Order/position within parent */
  order: z3.number().optional(),
});

/**
 * Schema for CopilotKit state properties
 */
const CopilotKitStateSchema = z3.object({
  copilotkit: z3
    .object({
      /** Context data from useCopilotReadable */
      context: z3.array(CopilotKitContextItemSchema).optional(),
    })
    .optional(),
  /** Planned nodes that the agent wants to create - frontend watches this */
  plannedNodes: z3.array(PlannedNodeSchema).nullable().optional(),
});

/**
 * TypeScript type for CopilotKit state
 */
export type CopilotKitState = z3.infer<typeof CopilotKitStateSchema>;

/**
 * TypeScript type for a planned node
 */
export type PlannedNode = z3.infer<typeof PlannedNodeSchema>;

/**
 * Options for the CopilotKit middleware
 */
export interface CopilotKitMiddlewareOptions {
  /**
   * Custom system prompt addition.
   */
  systemPromptAddition?: string;

  /**
   * Whether to include CopilotKit context in the system prompt.
   * @default true
   */
  includeContextInPrompt?: boolean;
}

/**
 * Default system prompt addition - tells agent to output structured JSON
 */
const DEFAULT_SYSTEM_PROMPT = `
## Node Creation

When asked to create nodes, content structures, or project hierarchies:
1. Research and plan the structure thoroughly
2. Output your planned nodes as structured JSON in your response
3. The frontend will automatically create the nodes based on your output

IMPORTANT: Do NOT attempt to call frontend tools directly. Instead, include your node plan in your response using this format:

\`\`\`json
{
  "plannedNodes": [
    { "id": "1", "title": "Module Name", "type": "module", "content": "Description..." },
    { "id": "2", "title": "Lesson Name", "type": "lesson", "parentId": "1", "content": "..." },
    { "id": "3", "title": "Topic Name", "type": "topic", "parentId": "2", "content": "..." }
  ]
}
\`\`\`

Use hierarchical IDs and parentId to establish the structure. The frontend will parse this JSON and create the nodes automatically.

For research and information gathering, use your available tools like internetSearch.
`;

/**
 * Format CopilotKit context for inclusion in system prompt
 */
function formatContextForPrompt(
  context: Array<{ description: string; value?: unknown }>,
): string {
  if (!context || context.length === 0) {
    return "";
  }

  const contextLines = context.map((item) => {
    const valueStr =
      typeof item.value === "object"
        ? JSON.stringify(item.value, null, 2)
        : String(item.value ?? "");
    return `### ${item.description}\n${valueStr}`;
  });

  return `
## Application Context

${contextLines.join("\n\n")}
`;
}

/**
 * Create CopilotKit middleware for deep agents.
 *
 * This middleware implements a state emission pattern:
 * - Agent researches and plans, then outputs structured JSON
 * - No frontend tool calls are made by the agent
 * - Frontend watches state and creates nodes based on agent output
 *
 * @example
 * ```typescript
 * const agent = createDeepAgent({
 *   model: "claude-sonnet-4-20250514",
 *   tools: [internetSearch],
 *   middleware: [createCopilotKitMiddleware()],
 * });
 * ```
 */
export function createCopilotKitMiddleware(
  options: CopilotKitMiddlewareOptions = {},
): Middleware {
  const {
    systemPromptAddition = DEFAULT_SYSTEM_PROMPT,
    includeContextInPrompt = true,
  } = options;

  return createMiddleware({
    name: "CopilotKitMiddleware",
    stateSchema: CopilotKitStateSchema,

    /**
     * Before model is called:
     * Add context and node creation instructions to the system prompt
     */
    beforeModel: (state: CopilotKitState) => {
      const copilotkit = state.copilotkit;
      const promptAdditions: string[] = [];

      // Add context information if available and enabled
      if (includeContextInPrompt && copilotkit?.context?.length) {
        const contextPrompt = formatContextForPrompt(copilotkit.context);
        if (contextPrompt) {
          promptAdditions.push(contextPrompt);
        }
      }

      // Add system prompt for node creation
      if (systemPromptAddition) {
        promptAdditions.push(systemPromptAddition);
      }

      if (promptAdditions.length > 0) {
        return {
          systemPrompt: promptAdditions.join("\n"),
        };
      }

      return undefined;
    },
  });
}

