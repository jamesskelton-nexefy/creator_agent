/**
 * Orchestrator Deep Agent
 *
 * Uses langchain's createAgent with middleware that dynamically injects
 * CopilotKit frontend tools into the model call.
 *
 * Architecture:
 * - Main Agent: createAgent with custom middleware
 * - Middleware captures CopilotKit tools and injects them via wrapModelCall
 * - Model can directly call CopilotKit frontend tools
 * - CopilotKit executes tool calls on the frontend
 * - Summarization middleware automatically condenses long conversations
 * - Long-term memory via LangGraph store with semantic search
 *
 * Key insight: wrapModelCall middleware can intercept model requests
 * and inject tools dynamically at runtime.
 */

import "dotenv/config";
import * as z from "zod";
import { MemorySaver, InMemoryStore } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createAgent, createMiddleware, summarizationMiddleware, tool, type ToolRuntime } from "langchain";

// ============================================================================
// CONTEXT SCHEMA FOR USER IDENTIFICATION
// ============================================================================

/**
 * Context schema for namespacing memories by user.
 * Pass userId when invoking the agent to scope memories to that user.
 */
const contextSchema = z.object({
  userId: z.string().optional(),
});

type ContextType = z.infer<typeof contextSchema>;

// ============================================================================
// MEMORY TOOL SCHEMAS
// ============================================================================

const SaveMemorySchema = z.object({
  content: z.string().describe("The content/information to remember"),
  category: z.string().optional().describe("Optional category to organize the memory (e.g., 'preferences', 'project-info', 'user-facts')"),
});

const RecallMemoriesSchema = z.object({
  query: z.string().describe("Search query to find relevant memories by semantic similarity"),
  limit: z.number().optional().describe("Maximum number of memories to return (default: 5)"),
});

const ListMemoriesSchema = z.object({
  category: z.string().optional().describe("Optional category to filter memories"),
  limit: z.number().optional().describe("Maximum number of memories to return (default: 10)"),
});

// ============================================================================
// MEMORY TOOLS
// ============================================================================

/**
 * Tool to save important information to long-term memory.
 * Memories persist across sessions and can be retrieved via semantic search.
 */
const saveMemory = tool(
  async (
    input: z.infer<typeof SaveMemorySchema>,
    runtime: ToolRuntime<unknown, ContextType>
  ) => {
    const userId = runtime.context?.userId ?? "default";
    const namespace = [userId, "memories"];
    const memoryId = `memory_${Date.now()}_${Math.random().toString(36).substring(7)}`;
    
    const memoryData = {
      content: input.content,
      category: input.category ?? "general",
      timestamp: new Date().toISOString(),
      createdAt: Date.now(),
    };
    
    await runtime.store.put(namespace, memoryId, memoryData);
    
    console.log(`[Memory] Saved memory for user ${userId}: ${input.content.substring(0, 50)}...`);
    
    return `Memory saved successfully. ID: ${memoryId}, Category: ${memoryData.category}`;
  },
  {
    name: "saveMemory",
    description: "Save important information to long-term memory. Use this to remember user preferences, important facts, project details, or any information that should persist across conversations.",
    schema: SaveMemorySchema,
  }
);

/**
 * Tool to search long-term memory by semantic similarity.
 * Uses embeddings to find memories related to the query.
 */
const recallMemories = tool(
  async (
    input: z.infer<typeof RecallMemoriesSchema>,
    runtime: ToolRuntime<unknown, ContextType>
  ) => {
    const userId = runtime.context?.userId ?? "default";
    const namespace = [userId, "memories"];
    const limit = input.limit ?? 5;
    
    const memories = await runtime.store.search(namespace, {
      query: input.query,
      limit,
    });
    
    console.log(`[Memory] Recalled ${memories.length} memories for user ${userId} matching: ${input.query}`);
    
    if (memories.length === 0) {
      return "No relevant memories found.";
    }
    
    const formattedMemories = memories.map((m, i) => {
      const value = m.value as { content: string; category: string; timestamp: string };
      return `${i + 1}. [${value.category}] ${value.content} (saved: ${value.timestamp})`;
    }).join("\n");
    
    return `Found ${memories.length} relevant memories:\n${formattedMemories}`;
  },
  {
    name: "recallMemories",
    description: "Search long-term memory for information related to a query. Uses semantic similarity to find relevant memories even if the exact words don't match.",
    schema: RecallMemoriesSchema,
  }
);

/**
 * Tool to list memories, optionally filtered by category.
 */
const listMemories = tool(
  async (
    input: z.infer<typeof ListMemoriesSchema>,
    runtime: ToolRuntime<unknown, ContextType>
  ) => {
    const userId = runtime.context?.userId ?? "default";
    const namespace = [userId, "memories"];
    const limit = input.limit ?? 10;
    
    // Search with empty query to get all, then filter by category if provided
    const searchOptions: { limit: number; filter?: Record<string, string> } = { limit };
    if (input.category) {
      searchOptions.filter = { category: input.category };
    }
    
    const memories = await runtime.store.search(namespace, searchOptions);
    
    console.log(`[Memory] Listed ${memories.length} memories for user ${userId}${input.category ? ` in category: ${input.category}` : ''}`);
    
    if (memories.length === 0) {
      return input.category 
        ? `No memories found in category: ${input.category}`
        : "No memories found.";
    }
    
    const formattedMemories = memories.map((m, i) => {
      const value = m.value as { content: string; category: string; timestamp: string };
      return `${i + 1}. [${value.category}] ${value.content}`;
    }).join("\n");
    
    return `Found ${memories.length} memories:\n${formattedMemories}`;
  },
  {
    name: "listMemories",
    description: "List stored memories, optionally filtered by category. Use to see what information has been saved.",
    schema: ListMemoriesSchema,
  }
);

// ============================================================================
// TOOL CATEGORIES FOR SUB-AGENT DELEGATION
// ============================================================================

/**
 * Core tools available to the main orchestrator.
 * These are always injected and handle navigation, UI, and user interaction.
 */
const CORE_ORCHESTRATOR_TOOLS = [
  // Navigation
  'navigateToProject',
  'goToProjectsList',
  'selectNode',
  'switchViewMode',
  // UI
  'toggleDetailPane',
  'showNotification',
  'expandNode',
  'collapseNode',
  // User Interaction
  'offerOptions',
  'askClarifyingQuestions',
  // Approvals
  'requestPlanApproval',
  'requestActionApproval',
  // Task Display
  'displayTodoList',
];

/**
 * Tool categories for sub-agent delegation.
 * Each sub-agent receives only its relevant tools.
 */
export const TOOL_CATEGORIES = {
  project: [
    'listProjects',
    'getProjectDetails',
    'createProject',
    'openProjectByName',
    'getProjectTemplates',
    'getClients',
  ],
  node: [
    'requestEditMode',
    'releaseEditMode',
    'checkEditStatus',
    'getProjectHierarchyInfo',
    'getNodeChildren',
    'getNodeDetails',
    'getNodesByLevel',
    'getAvailableTemplates',
    'getNodeTemplateFields',
    'getNodeFields',
    'createNode',
    'updateNodeFields',
    'scrollToNode',
    'expandAllNodes',
    'collapseAllNodes',
    // Template tools merged into node agent
    'listAllNodeTemplates',
    'listFieldTemplates',
    'getTemplateDetails',
  ],
  data: [
    'switchTableViewMode',
    'getTableViewState',
    'addTableFilter',
    'clearTableFilters',
    'addTableSort',
    'clearTableSorts',
    'setTableGrouping',
    'clearTableGrouping',
    'setTableColumnVisibility',
    'exportTableData',
    'saveTableView',
    'loadTableView',
  ],
  document: [
    'uploadDocument',
    'listDocuments',
    'searchDocuments',
    'searchDocumentsByText',
    'getDocumentByName',
    'getDocumentLines',
  ],
  media: [
    'searchMicroverse',
    'getMicroverseDetails',
    'getMicroverseUsage',
    'generateMicroverseAssets',
    'attachMicroverseToNode',
    'detachMicroverseFromNode',
    'getNodeMicroverseFields',
  ],
  framework: [
    'listFrameworks',
    'getFrameworkDetails',
    'searchASQAUnits',
    'getUnitDetails',
    'importFramework',
    'linkFrameworkToProject',
    'mapCriteriaToNode',
    'suggestCriteriaMappings',
  ],
  memory: [
    'saveMemory',
    'recallMemories',
    'listMemories',
  ],
};

/**
 * Get tool name from CopilotKit tool object
 */
function getToolName(tool: any): string {
  return tool.name || tool.function?.name || '';
}

/**
 * Filter tools to only include those in the allowed list
 */
function filterTools(tools: any[], allowedNames: string[]): any[] {
  return tools.filter((tool) => allowedNames.includes(getToolName(tool)));
}

// ============================================================================
// COPILOTKIT TOOLS INJECTION MIDDLEWARE
// ============================================================================

/**
 * Get tools for a specific category
 */
function getToolsForCategory(allTools: any[], category: keyof typeof TOOL_CATEGORIES): any[] {
  const categoryToolNames = TOOL_CATEGORIES[category];
  return filterTools(allTools, categoryToolNames);
}

/**
 * Detect if the last message requests additional tools via [NEED:category] marker
 */
function detectToolRequest(messages: any[]): (keyof typeof TOOL_CATEGORIES)[] {
  if (!messages || messages.length === 0) return [];
  
  const lastMsg = messages[messages.length - 1];
  const content = typeof lastMsg.content === 'string' 
    ? lastMsg.content 
    : Array.isArray(lastMsg.content)
      ? lastMsg.content.filter((b: any) => b.type === 'text').map((b: any) => b.text).join('\n')
      : '';
  
  const categories: (keyof typeof TOOL_CATEGORIES)[] = [];
  
  // Look for [NEED:category] markers
  const needPattern = /\[NEED:(\w+)\]/gi;
  let match;
  while ((match = needPattern.exec(content)) !== null) {
    const cat = match[1].toLowerCase() as keyof typeof TOOL_CATEGORIES;
    if (cat in TOOL_CATEGORIES) {
      categories.push(cat);
    }
  }
  
  return categories;
}

/**
 * Middleware that captures CopilotKit frontend tools from state and
 * injects ONLY the core orchestrator tools into model calls.
 * 
 * Sub-agents receive their specific tool subsets via delegation.
 * This prevents payload overflow from 70+ tools being sent at once.
 * 
 * Tool expansion: When the orchestrator needs specialized tools, it can
 * include [NEED:category] in its response (e.g., [NEED:node], [NEED:project]).
 * The middleware will then expand the available tools on the next invocation.
 * 
 * Key insight: By setting tools: [] on createAgent and injecting CopilotKit tools
 * via wrapModelCall, we bypass ToolNode entirely. All tool calls go to END
 * and CopilotKit handles execution on the frontend.
 */
const copilotKitToolsMiddleware = createMiddleware({
  name: "CopilotKitToolsInjection",
  stateSchema: z.object({
    // CopilotKit populates this when invoking the agent
    copilotkit: z.object({
      actions: z.array(z.any()).optional(),
      context: z.array(z.any()).optional(),
    }).optional(),
    // Internal: store captured tools (filtered to core set)
    _copilotKitTools: z.array(z.any()).optional(),
    // Store ALL tools for sub-agent delegation
    _allCopilotKitTools: z.array(z.any()).optional(),
    // Track requested tool categories
    _requestedCategories: z.array(z.string()).optional(),
    // Messages for tool request detection
    messages: z.array(z.any()).optional(),
  }),
  
  // Capture and filter CopilotKit tools before each model call
  beforeModel: (state) => {
    const allTools = (state as any).copilotkit?.actions ?? [];
    const messages = (state as any).messages ?? [];
    
    // Check if previous response requested additional tools
    const requestedCategories = detectToolRequest(messages);
    
    // Start with core tools
    let toolsToInject = filterTools(allTools, CORE_ORCHESTRATOR_TOOLS);
    
    // Add tools from requested categories
    if (requestedCategories.length > 0) {
      console.log(`[CopilotKitToolsMiddleware] Expanding tools for categories: ${requestedCategories.join(', ')}`);
      
      for (const category of requestedCategories) {
        const categoryTools = getToolsForCategory(allTools, category);
        toolsToInject = [...toolsToInject, ...categoryTools];
      }
      
      // Deduplicate
      const seenNames = new Set<string>();
      toolsToInject = toolsToInject.filter((t: any) => {
        const name = getToolName(t);
        if (seenNames.has(name)) return false;
        seenNames.add(name);
        return true;
      });
    }
    
    console.log(`[CopilotKitToolsMiddleware] Captured ${allTools.length} frontend tools, injecting ${toolsToInject.length}`);
    
    if (toolsToInject.length > 0) {
      const toolNames = toolsToInject.slice(0, 10).map((t: any) => getToolName(t)).join(", ");
      console.log(`[CopilotKitToolsMiddleware] Tools: ${toolNames}${toolsToInject.length > 10 ? '...' : ''}`);
    }
    
    return {
      _copilotKitTools: toolsToInject,
      _allCopilotKitTools: allTools,
      _requestedCategories: requestedCategories,
    };
  },

  // Inject filtered CopilotKit tools into the model call
  wrapModelCall: async (request, handler) => {
    // Get filtered tools from state
    const toolsToInject = (request as any).state?._copilotKitTools ?? 
                          (request as any).runtime?.state?._copilotKitTools ?? [];
    
    console.log(`[CopilotKitToolsMiddleware] Injecting ${toolsToInject.length} tools into model call`);
    
    if (toolsToInject.length > 0) {
      // Get existing tools from the request (should be empty since tools: [])
      const existingTools = (request as any).tools ?? [];
      const finalTools = [...existingTools, ...toolsToInject];
      
      console.log(`[CopilotKitToolsMiddleware] Total tools: ${finalTools.length}`);
      
      // Create modified request with filtered tools
      try {
        if (typeof request.replace === 'function') {
          return handler(request.replace({ tools: finalTools }));
        } else {
          (request as any).tools = finalTools;
          return handler(request);
        }
      } catch (error) {
        console.error(`[CopilotKitToolsMiddleware] Error injecting tools:`, error);
        return handler(request);
      }
    }
    
    return handler(request);
  },
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const ORCHESTRATOR_SYSTEM_PROMPT = `You are The Orchestrator Deep - a powerful AI assistant for creating impactful online training content.

## Architecture

You are the main orchestrator with access to core tools for navigation and user interaction.
For specialized operations, you delegate to sub-agents who have their own tool sets.

## Your Core Tools (Direct Access)

### Navigation
- **navigateToProject(projectId)** - Navigate to a specific project
- **goToProjectsList()** - Return to the projects list
- **selectNode(nodeId)** - Select a node in the tree
- **switchViewMode(mode)** - Change view: "document", "list", "graph", or "table"

### UI Control
- **toggleDetailPane()** - Show/hide the detail panel
- **showNotification(message, type)** - Show a toast notification
- **expandNode(nodeId)** / **collapseNode(nodeId)** - Expand/collapse tree nodes

### User Interaction
- **offerOptions(question, options)** - Present choices to user with clickable buttons
- **askClarifyingQuestions(questions)** - Ask sequential questions with options

### Approvals
- **requestPlanApproval(plan)** - Get user approval for a plan before execution
- **requestActionApproval(action)** - Confirm a sensitive action

### Task Display
- **displayTodoList(title?, todos)** - Show task progress in chat

## Specialized Sub-Agents (Delegation)

For specialized operations, describe what you need and the appropriate sub-agent will handle it:

### Project Agent
Handles: listing projects, project details, creating projects, finding clients
Use when: User asks about projects, wants to create/open a project

### Node Agent  
Handles: edit mode, creating/updating nodes, hierarchy info, templates, node queries
Use when: User wants to create content, edit nodes, explore structure

### Data Agent
Handles: table view operations, filtering, sorting, grouping, exports
Use when: User wants to work with table view or query data

### Document Agent
Handles: document uploads, searching documents (RAG), listing documents
Use when: User wants to search uploaded documents or upload new ones

### Media Agent
Handles: media library (Microverse), attaching/detaching media to nodes
Use when: User wants to work with images, videos, or other media assets

### Framework Agent
Handles: competency frameworks, ASQA units, criteria mapping
Use when: User wants to work with training frameworks or compliance

### Memory Agent
Handles: saving/recalling/listing long-term memories
Use when: User asks to remember something or wants to recall past information

## CRITICAL: Call tools to take action

When the user asks you to do something within your core capabilities, CALL the appropriate tool immediately.

### Examples:
- "switch to list view" → CALL switchViewMode({ mode: "list" })
- "go to graph view" → CALL switchViewMode({ mode: "graph" })
- "select the first module" → CALL selectNode({ nodeId: "..." })
- "show me my options" → CALL offerOptions({ question: "What would you like to do?", options: [...] })

## Requesting Specialized Tools

When you need tools from a specialized category, include the marker [NEED:category] in your response.
The system will then provide those tools on the next turn.

**Available categories:**
- [NEED:project] - Project listing, creation, navigation
- [NEED:node] - Node creation, editing, templates, hierarchy
- [NEED:data] - Table view, filtering, sorting, exports
- [NEED:document] - Document search (RAG), uploads
- [NEED:media] - Media library (Microverse), attachments
- [NEED:framework] - Competency frameworks, criteria mapping
- [NEED:memory] - Long-term memory save/recall

**Example:**
User: "Create a new module called Safety Training"
You: "I'll help you create that module. [NEED:node]"
(Next turn, you'll have access to node tools like requestEditMode, createNode, etc.)

## Communication Style

1. Be conversational and helpful
2. Execute tool calls immediately when within your capabilities
3. For specialized tasks, request the tool category you need
4. Summarize results and offer next steps
5. Use offerOptions to present clear choices to users`;


// ============================================================================
// STORE SETUP (Long-Term Memory)
// ============================================================================

console.log("[orchestrator-deep] Setting up long-term memory store...");

/**
 * Initialize the memory store.
 * - On LangGraph Platform: Uses managed store (automatically provided)
 * - Local development: Uses InMemoryStore with OpenAI embeddings
 */
let store: InMemoryStore;

// Check if running on LangGraph Platform (store will be injected automatically)
// For local dev, create InMemoryStore with embeddings for semantic search
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (OPENAI_API_KEY) {
  console.log("[orchestrator-deep] Initializing InMemoryStore with OpenAI embeddings...");
  store = new InMemoryStore({
    index: {
      embeddings: new OpenAIEmbeddings({ 
        model: "text-embedding-3-small",
        openAIApiKey: OPENAI_API_KEY,
      }),
      dims: 1536,
    },
  });
  console.log("[orchestrator-deep] InMemoryStore initialized with semantic search");
} else {
  console.log("[orchestrator-deep] No OPENAI_API_KEY found, using InMemoryStore without embeddings...");
  console.log("[orchestrator-deep] WARNING: Semantic search will not work without embeddings");
  store = new InMemoryStore();
}

// ============================================================================
// CHECKPOINTER SETUP (Short-Term Memory / Thread Persistence)
// ============================================================================

console.log("[orchestrator-deep] Setting up checkpointer...");

const SUPABASE_DB_URL = process.env.SUPABASE_DB_URL;

let checkpointer: PostgresSaver | MemorySaver;

if (SUPABASE_DB_URL) {
  console.log("[orchestrator-deep] Initializing PostgreSQL checkpointer...");
  checkpointer = PostgresSaver.fromConnString(SUPABASE_DB_URL);
  await checkpointer.setup();
  console.log("[orchestrator-deep] PostgreSQL checkpointer initialized");
} else {
  console.log("[orchestrator-deep] Initializing MemorySaver checkpointer for local dev...");
  checkpointer = new MemorySaver();
  console.log("[orchestrator-deep] MemorySaver checkpointer initialized");
}

// ============================================================================
// CREATE AGENT
// ============================================================================

console.log("[orchestrator-deep] Creating agent...");

// Create the agent with middleware that injects CopilotKit tools and handles summarization
// NOTE: CopilotKit tools are injected via wrapModelCall to bypass ToolNode.
// Memory tools cannot be added this way (LangChain validation blocks it).
// All tool calls go to END -> CopilotKit executes on frontend.
export const agent = createAgent({
  model: "claude-sonnet-4-20250514",
  systemPrompt: ORCHESTRATOR_SYSTEM_PROMPT,
  tools: [], // NO tools - CopilotKit tools injected via wrapModelCall
  middleware: [
    // CopilotKit tools injection - captures and injects frontend tools
    copilotKitToolsMiddleware,
    // NOTE: todoListMiddleware is NOT used here because it conflicts with CopilotKit tools.
    // When middleware adds tools, LangGraph routes to ToolNode which doesn't know CopilotKit actions.
    // Instead, we use the frontend displayTodoList action for todo display in chat.
    // Summarization middleware - automatically condenses long conversations
    // Triggers when conversation exceeds 8000 tokens, keeps last 20 messages
    summarizationMiddleware({
      model: "claude-sonnet-4-20250514",
      trigger: { tokens: 8000 },
      keep: { messages: 20 },
    }),
  ],
  contextSchema,
  store,
  checkpointer,
});

console.log("[orchestrator-deep] Agent created successfully");
console.log("[orchestrator-deep] Pattern: createAgent + wrapModelCall CopilotKit tools injection");
console.log("[orchestrator-deep] Tools: Dynamically injected from CopilotKit state (frontend tools only)");
console.log("[orchestrator-deep] TodoList: displayTodoList frontend action for task planning display");
console.log("[orchestrator-deep] Summarization: Triggers at 8000 tokens, keeps 20 messages");
