/**
 * Orchestrator Agent - LangChain createAgent Version
 *
 * A simpler orchestrator using LangChain's standard createAgent approach
 * instead of manual LangGraph state management. This may provide more
 * stable tool handling and routing.
 *
 * This agent:
 * - Uses createAgent from langchain v1 (built on LangGraph internally)
 * - Has access to all frontend tools via CopilotKit state
 * - Provides a simpler, more standard agent loop
 * - Uses custom middleware to intercept frontend tools for CopilotKit auto-execution
 */

import "dotenv/config";
import { createAgent, createMiddleware } from "langchain";
import { SystemMessage, ToolMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { CopilotKitStateAnnotation, convertActionsToDynamicStructuredTools, copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";

// ============================================================================
// STATE DEFINITION
// ============================================================================

/**
 * Simple state annotation extending CopilotKit state.
 * The createAgent handles most state internally, we just need
 * to pass through the CopilotKit context and actions.
 */
const OrchestratorLangchainStateAnnotation = Annotation.Root({
  // Inherit CopilotKit state (messages, actions, context)
  ...CopilotKitStateAnnotation.spec,
});

export type OrchestratorLangchainState = typeof OrchestratorLangchainStateAnnotation.State;

// ============================================================================
// FRONTEND TOOL MIDDLEWARE - Intercept frontend tools for CopilotKit execution
// ============================================================================

/**
 * Creates middleware that intercepts frontend tool calls and short-circuits
 * server-side execution. Instead, the tool call is emitted to CopilotKit
 * which executes it client-side in the browser.
 * 
 * This follows the LangChain custom middleware pattern:
 * https://docs.langchain.com/oss/javascript/langchain/middleware/custom#wrap-style-hooks
 * 
 * How it works:
 * 1. Agent calls a frontend tool (e.g., switchViewMode)
 * 2. Middleware intercepts via wrapToolCall
 * 3. Middleware returns placeholder ToolMessage (doesn't execute server-side)
 * 4. Tool call is emitted to CopilotKit (via emitToolCalls: true)
 * 5. CopilotKit executes the tool in the browser
 * 6. CopilotKit sends real ToolMessage result back
 * 7. Agent continues with actual result
 */
function createFrontendToolMiddleware(frontendToolNames: Set<string>) {
  return createMiddleware({
    name: "FrontendToolMiddleware",
    wrapToolCall: (request, handler) => {
      // Check if this is a frontend tool that should be executed client-side
      if (frontendToolNames.has(request.toolCall.name)) {
        console.log(`[FrontendToolMiddleware] Intercepting frontend tool: ${request.toolCall.name}`);
        console.log(`[FrontendToolMiddleware] Args:`, JSON.stringify(request.toolCall.args, null, 2));
        console.log(`[FrontendToolMiddleware] Skipping server execution - CopilotKit will handle this client-side`);
        
        // Return placeholder ToolMessage - CopilotKit will execute and provide real result
        // The tool call is still emitted (via emitToolCalls: true) so CopilotKit sees it
        return new ToolMessage({
          content: JSON.stringify({ 
            pending: true, 
            executedBy: "copilotkit",
            note: "This tool is executed client-side by CopilotKit" 
          }),
          tool_call_id: request.toolCall.id,
          name: request.toolCall.name,
        });
      }
      
      // For backend tools, execute normally on the server
      console.log(`[FrontendToolMiddleware] Executing backend tool: ${request.toolCall.name}`);
      return handler(request);
    },
  });
}

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const ORCHESTRATOR_SYSTEM_PROMPT = `You are The Orchestrator - a helpful AI assistant for creating impactful online training content.

## Your Capabilities

You have access to frontend tools that allow you to:

### Navigation & View Control
- **navigateToProject(projectId)** - Navigate to a specific project by ID
- **goToProjectsList()** - Return to the projects list page
- **switchViewMode(mode)** - Change view: "document", "list", "graph", or "table"
- **selectNode(nodeId)** - Select a node in the tree
- **scrollToNode(nodeId)** - Scroll to make a node visible
- **expandNode(nodeId)** - Expand a collapsed node
- **collapseNode(nodeId)** - Collapse an expanded node
- **expandAllNodes()** - Expand entire tree
- **collapseAllNodes()** - Collapse entire tree
- **toggleDetailPane()** - Show/hide the detail panel
- **showNotification(message, type)** - Show a toast notification

### Edit Mode (REQUIRED before creating/updating nodes)
- **requestEditMode()** - Request edit lock before creating/updating nodes
- **releaseEditMode()** - Release the edit lock when done editing
- **checkEditStatus()** - Check if you have edit access

### Project Management
- **listProjects(searchTerm?, clientId?, sortBy?)** - List projects
- **getProjectDetails(projectId?, projectName?)** - Get project info
- **createProject(name, clientId, description?, templateId?)** - Create new project
- **openProjectByName(projectName)** - Search and navigate to project
- **getProjectTemplates()** - List available project templates
- **getClients()** - List available clients

### Node Information (Read-only)
- **getProjectHierarchyInfo()** - Get hierarchy levels, coding config
- **getNodeChildren(nodeId?)** - Get children of a node
- **getNodeDetails(nodeId?)** - Get detailed info about a node
- **getNodesByLevel(level?, levelName?, limit?)** - Find nodes at a hierarchy level
- **getAvailableTemplates(parentNodeId?)** - Get templates valid for parent
- **getNodeTemplateFields(templateId)** - Get field schema for a template
- **getNodeFields(nodeId?)** - Read current field values

### Node Creation & Editing (REQUIRES edit mode)
- **createNode(templateId, title, parentNodeId?, initialFields?)** - Create a node
- **updateNodeFields(nodeId?, fieldUpdates)** - Update fields

### Template Exploration
- **listAllNodeTemplates(projectTemplateId?, nodeType?, limit?)** - Browse node templates
- **listFieldTemplates(fieldType?, limit?)** - Browse field templates
- **getTemplateDetails(templateId, templateType?)** - Get full template details

### Document Management
- **uploadDocument(category, instructions?)** - Trigger upload dialog

### Media Library (Microverse)
- **searchMicroverse(query?, fileType?, category?, limit?)** - Search media
- **getMicroverseDetails(fileId)** - Get media asset info
- **getMicroverseUsage(fileId)** - Check where asset is used
- **attachMicroverseToNode(nodeId, fileId, fieldAssignmentId)** - Attach media
- **detachMicroverseFromNode(nodeId, fileId, fieldAssignmentId)** - Remove media

### Framework & Criteria Mapping
- **listFrameworks(category?, status?)** - List competency frameworks
- **getFrameworkDetails(frameworkId)** - Get framework info
- **searchASQAUnits(query, limit?)** - Search ASQA training units
- **linkFrameworkToProject(frameworkId, projectId)** - Link framework
- **mapCriteriaToNode(nodeId, criteriaId)** - Map criteria to node
- **suggestCriteriaMappings(nodeId)** - Get AI-suggested mappings

### User Interaction Tools
- **askClarifyingQuestions(questions)** - Ask questions with options
- **offerOptions(title, options, allowMultiple?)** - Present choices to user
- **requestPlanApproval(plan, title?)** - Get approval for a plan
- **requestActionApproval(action, reason?)** - Confirm sensitive action

## Guidelines

1. **Always ask before acting** - Present options and wait for user direction
2. **Use tools directly** when possible for navigation, listing, creating nodes
3. **Use offerOptions or askClarifyingQuestions** when presenting choices
4. **Request edit mode** before creating or updating any nodes
5. **Be helpful and concise** in your responses
6. **Summarize what was done** after completing any action

## Important Rules

- ALWAYS call frontend tools ONE AT A TIME, never in parallel
- After ANY action, summarize what was done and offer next steps
- If unsure, ask the user for clarification
- Keep the user informed of progress`;

// ============================================================================
// AGENT NODE - Uses LangChain createAgent with custom middleware
// ============================================================================

async function orchestratorNode(
  state: OrchestratorLangchainState,
  config: RunnableConfig
): Promise<Partial<OrchestratorLangchainState>> {
  console.log("\n[orchestrator-langchain] ============ Agent Node ============");
  console.log("  Messages count:", state.messages?.length ?? 0);

  // Get frontend actions from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  console.log("  Available frontend actions:", frontendActions.length);
  
  // Build set of frontend tool names for the middleware
  const frontendToolNames = new Set(
    frontendActions.map((action: any) => {
      const a = action.type === "function" ? action.function : action;
      return a.name;
    })
  );
  console.log("  Frontend tool names:", Array.from(frontendToolNames).join(", ") || "none");

  // Convert CopilotKit actions to DynamicStructuredTools
  // These tools don't need to do anything server-side - the middleware handles them
  const frontendTools = convertActionsToDynamicStructuredTools(frontendActions);
  console.log("  Converted to tools:", frontendTools.map(t => t.name).join(", ") || "none");

  // Customize config to emit tool calls for CopilotKit
  // This is essential - it tells CopilotKit about the tool calls so it can execute them
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
  });

  // Create the agent using LangChain's createAgent with custom middleware
  // The middleware intercepts frontend tools and short-circuits server execution
  const agent = createAgent({
    model: "anthropic:claude-sonnet-4-20250514",
    tools: frontendTools,
    systemPrompt: new SystemMessage({
      content: [
        {
          type: "text",
          text: ORCHESTRATOR_SYSTEM_PROMPT,
          // Enable Anthropic prompt caching for the static system prompt
          cache_control: { type: "ephemeral" },
        },
      ],
    }),
    middleware: [createFrontendToolMiddleware(frontendToolNames)],
  });

  console.log("  Invoking createAgent with FrontendToolMiddleware...");

  try {
    // Invoke the agent with the current messages
    const result = await agent.invoke(
      { messages: state.messages || [] },
      customConfig
    );

    console.log("  Agent response received");
    console.log("  Output messages:", result.messages?.length || 0);

    return {
      messages: result.messages,
    };
  } catch (error) {
    console.error("  [ERROR] Agent invocation failed:", error);
    throw error;
  }
}

// ============================================================================
// ROUTING - Simple routing for the single-node agent
// ============================================================================

function shouldContinue(state: OrchestratorLangchainState): "__end__" {
  // createAgent handles its own tool execution loop internally
  // We always route to END after the agent completes
  console.log("\n[routing] Agent completed, routing to END");
  return "__end__";
}

// ============================================================================
// GRAPH DEFINITION
// ============================================================================

console.log("[orchestrator-langchain] Building workflow graph...");

const workflow = new StateGraph(OrchestratorLangchainStateAnnotation)
  .addNode("orchestrator", orchestratorNode)
  .addEdge(START, "orchestrator")
  .addConditionalEdges("orchestrator", shouldContinue, {
    __end__: END,
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
console.log("[orchestrator-langchain] Using LangChain createAgent with FrontendToolMiddleware");
