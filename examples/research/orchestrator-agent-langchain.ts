/**
 * Orchestrator Agent - LangChain Simple Pattern
 *
 * Uses model.bindTools() with StateGraph (not createAgent).
 * CopilotKit handles frontend tool execution naturally via tool_calls routing.
 * 
 * Flow:
 * 1. Model receives tools from CopilotKit state
 * 2. Model generates response (text or tool_calls)
 * 3. If tool_calls → route to END → CopilotKit executes client-side
 * 4. CopilotKit resumes with tool results
 */

import "dotenv/config";
import { ChatAnthropic } from "@langchain/anthropic";
import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { 
  CopilotKitStateAnnotation, 
  copilotkitCustomizeConfig
} from "@copilotkit/sdk-js/langgraph";

// ============================================================================
// STATE DEFINITION
// ============================================================================

const OrchestratorStateAnnotation = Annotation.Root({
  ...CopilotKitStateAnnotation.spec,
});

export type OrchestratorState = typeof OrchestratorStateAnnotation.State;

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const ORCHESTRATOR_SYSTEM_PROMPT = `You are The Orchestrator - a helpful AI assistant for creating impactful online training content.

## Your Capabilities

You have access to frontend tools that allow you to:

### Navigation & View Control
- **switchViewMode(mode)** - Change view: "document", "list", "graph", or "table"
- **navigateToProject(projectId)** - Navigate to a specific project by ID
- **goToProjectsList()** - Return to the projects list page
- **selectNode(nodeId)** - Select a node in the tree
- **toggleDetailPane()** - Show/hide the detail panel
- **showNotification(message, type)** - Show a toast notification

### Project Management
- **listProjects(searchTerm?, clientId?)** - List projects
- **getProjectDetails(projectId?, projectName?)** - Get project info
- **createProject(name, clientId, description?, templateId?)** - Create new project
- **openProjectByName(projectName)** - Search and navigate to project

### Node Operations
- **getNodeDetails(nodeId?)** - Get detailed info about a node
- **getNodeChildren(nodeId?)** - Get children of a node
- **createNode(templateId, title, parentNodeId?, initialFields?)** - Create a node
- **updateNodeFields(nodeId?, fieldUpdates)** - Update fields

### User Interaction
- **offerOptions(question, option_1, option_2?, option_3?)** - Present choices to user
- **askClarifyingQuestions(questions)** - Ask questions with options

## CRITICAL: You MUST use tools to take action

When the user asks you to do something, you MUST call the appropriate tool. DO NOT just say you will do it.

Examples:
- "switch to graph view" → CALL switchViewMode({ mode: "graph" })
- "show me projects" → CALL listProjects({})
- "go to document view" → CALL switchViewMode({ mode: "document" })

ALWAYS call the tool. NEVER just describe what you would do.`;

// ============================================================================
// MODEL SETUP
// ============================================================================

const model = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  temperature: 0,
});

// ============================================================================
// AGENT NODE - Simple model.bindTools() pattern
// ============================================================================

async function orchestratorNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[orchestrator-langchain] ============ Agent Node ============");
  console.log("  Messages count:", state.messages?.length ?? 0);

  // Get frontend tools from CopilotKit state
  const tools = state.copilotkit?.actions ?? [];
  console.log("  Frontend tools available:", tools.length);

  // Customize config to emit tool calls for CopilotKit
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
  });

  // Bind tools to model (CopilotKit actions are already in tool format)
  const modelWithTools = model.bindTools(tools);

  // Build messages with system prompt
  const messages: BaseMessage[] = [
    { role: "system", content: ORCHESTRATOR_SYSTEM_PROMPT } as any,
    ...(state.messages || []),
  ];

  console.log("  Calling model with", tools.length, "bound tools...");

  try {
    const response = await modelWithTools.invoke(messages, customConfig);

    // Log what we got
    const aiMessage = response as AIMessage;
    if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
      console.log("  Tool calls:", aiMessage.tool_calls.map(tc => tc.name).join(", "));
    } else {
      console.log("  Text response (no tool calls)");
    }

    return {
      messages: [response],
    };
  } catch (error) {
    console.error("  [ERROR] Model invocation failed:", error);
    throw error;
  }
}

// ============================================================================
// ROUTING - Handle tool execution flow
// ============================================================================

function shouldContinue(state: OrchestratorState): "__end__" | "orchestrator" {
  const messages = state.messages || [];
  const lastMessage = messages[messages.length - 1];
  
  if (!lastMessage) {
    console.log("[routing] No messages, ending");
    return "__end__";
  }

  // Check if last message is a ToolMessage (result from CopilotKit)
  // If so, continue to the agent to process the result
  if ('tool_call_id' in lastMessage) {
    console.log("[routing] ToolMessage received → continue to agent");
    return "orchestrator";
  }

  // Check if last message is AIMessage with tool calls
  const aiMessage = lastMessage as AIMessage;
  if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
    console.log("[routing] Tool calls present → END (CopilotKit executes)");
    return "__end__";
  }
  
  // No tool calls, we're done
  console.log("[routing] No tool calls → END");
  return "__end__";
}

// ============================================================================
// GRAPH DEFINITION
// ============================================================================

console.log("[orchestrator-langchain] Building workflow graph...");

const workflow = new StateGraph(OrchestratorStateAnnotation)
  .addNode("orchestrator", orchestratorNode)
  .addEdge(START, "orchestrator")
  .addConditionalEdges("orchestrator", shouldContinue, {
    __end__: END,
    orchestrator: "orchestrator",
  });

// ============================================================================
// PERSISTENCE SETUP
// ============================================================================

const SUPABASE_DB_URL = process.env.SUPABASE_DB_URL;
const IS_LANGSMITH_CLOUD = !SUPABASE_DB_URL;

let checkpointer: PostgresSaver | undefined;

if (!IS_LANGSMITH_CLOUD && SUPABASE_DB_URL) {
  console.log("[orchestrator-langchain] Initializing PostgreSQL checkpointer...");
  console.log("[orchestrator-langchain] DB URL:", SUPABASE_DB_URL.replace(/:[^:@]+@/, ":****@"));

  checkpointer = PostgresSaver.fromConnString(SUPABASE_DB_URL);
  await checkpointer.setup();
  console.log("[orchestrator-langchain] PostgreSQL checkpointer initialized");
} else {
  console.log("[orchestrator-langchain] Running in LangSmith Cloud - using managed checkpointer");
}

// Compile the graph
export const agent = workflow.compile({
  ...(checkpointer && { checkpointer }),
});

console.log("[orchestrator-langchain] Workflow graph compiled successfully");
console.log("[orchestrator-langchain] Pattern: model.bindTools() + StateGraph");
console.log("[orchestrator-langchain] Tool execution: CopilotKit handles client-side");
