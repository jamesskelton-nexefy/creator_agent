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
 * - Uses interrupt() for frontend tools so CopilotKit can execute them
 */

import "dotenv/config";
import { createAgent, createMiddleware } from "langchain";
import { SystemMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { CopilotKitStateAnnotation } from "@copilotkit/sdk-js/langgraph";
import { copilotkitCustomizeConfig, copilotKitInterrupt } from "@copilotkit/sdk-js/langgraph";

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
// AGENT NODE - Uses LangChain createAgent internally
// ============================================================================

async function orchestratorNode(
  state: OrchestratorLangchainState,
  config: RunnableConfig
): Promise<Partial<OrchestratorLangchainState>> {
  console.log("\n[orchestrator-langchain] ============ Agent Node ============");
  console.log("  Messages count:", state.messages?.length ?? 0);

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  console.log("  Available frontend tools:", frontendActions.length);
  
  // Build a set of frontend tool names for quick lookup
  const frontendToolNames = new Set(
    frontendActions.map((t: { name: string }) => t.name)
  );
  console.log("  Tool names:", Array.from(frontendToolNames).join(", ") || "none");

  // Customize config to emit tool calls for CopilotKit
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
  });

  // Create middleware to intercept frontend tool calls and use copilotKitInterrupt
  const frontendToolMiddleware = createMiddleware({
    name: "FrontendToolInterrupt",
    wrapToolCall: async (request, handler) => {
      const toolName = request.toolCall.name;
      const toolArgs = request.toolCall.args;
      
      // Check if this is a frontend tool (all CopilotKit actions are frontend tools)
      if (frontendToolNames.has(toolName)) {
        console.log(`\n[middleware] Frontend tool detected: ${toolName}`);
        console.log(`[middleware] Args:`, JSON.stringify(toolArgs, null, 2));
        console.log(`[middleware] Using copilotKitInterrupt for CopilotKit to execute...`);
        
        // Use copilotKitInterrupt to pause the graph with the proper CopilotKit format
        // CopilotKit will receive the tool call, execute it in the browser,
        // and resume the graph with the result
        const { answer } = copilotKitInterrupt({
          action: toolName,
          args: toolArgs as Record<string, any>,
        });
        
        console.log(`[middleware] Resumed from interrupt with answer:`, answer);
        
        // Return the answer from the interrupt (CopilotKit provides this when resuming)
        return answer;
      }
      
      // For non-frontend tools, execute normally via the handler
      console.log(`[middleware] Server tool: ${toolName}, executing normally`);
      return await handler(request);
    },
  });

  // Create the agent using LangChain's createAgent with middleware array
  const agent = createAgent({
    model: "anthropic:claude-sonnet-4-20250514",
    tools: frontendActions,
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
    // Middleware must be an array
    middleware: [frontendToolMiddleware],
  });

  console.log("  Invoking createAgent with frontend tool middleware...");

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
console.log("[orchestrator-langchain] Using LangChain createAgent pattern");

