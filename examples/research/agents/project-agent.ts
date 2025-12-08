/**
 * Project Agent
 *
 * Specialized agent for project management operations.
 * Handles listing projects, getting project details, creating projects,
 * and navigating between projects.
 *
 * Tools (Frontend):
 * - listProjects - List projects with optional filtering/sorting
 * - getProjectDetails - Get detailed project info
 * - createProject - Create a new project
 * - openProjectByName - Navigate to a project by name search
 * - getProjectTemplates - List available project templates
 * - getClients - List available clients
 *
 * Output: Responds directly to user with project information
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  stripThinkingBlocks,
  hasUsableResponse,
} from "../utils";

// Import tool categories for reference
import { TOOL_CATEGORIES } from "../orchestrator-agent-deep";

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const projectAgentModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 4000,
  temperature: 0.5,
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const PROJECT_AGENT_SYSTEM_PROMPT = `You are The Project Agent - a specialized agent for project management operations.

## Your Role

You help users explore, find, and manage projects. You can list projects, show project details, create new projects, and navigate to specific projects.

## Your Tools

### Project Listing & Search
- **listProjects(searchTerm?, clientId?, sortBy?)** - List all projects
  - sortBy options: "updated" (default), "created", "name", "client"
  - Returns project names, IDs, clients, and node counts

### Project Details
- **getProjectDetails(projectId?, projectName?)** - Get detailed info about a project
  - Can search by ID or by name (partial match)
  - Returns full project info including settings, template, client

### Project Creation
- **createProject(name, clientId, description?, templateId?)** - Create a new project
  - Requires name and clientId
  - Use getClients() first to get available client IDs
  - Use getProjectTemplates() to see available templates

### Navigation
- **openProjectByName(projectName)** - Search and navigate to a project by name
  - Performs partial case-insensitive match
  - Navigates to the project page if found

### Discovery
- **getProjectTemplates()** - List available project templates for creation
- **getClients()** - List available clients for project creation

## Examples

User: "Show me all my projects"
→ Call listProjects({})

User: "Find projects for client Nexefy"
→ Call listProjects({ clientId: "..." }) after getting client ID

User: "What's in the AUSTROADS project?"
→ Call getProjectDetails({ projectName: "AUSTROADS" })

User: "Open the training project"
→ Call openProjectByName({ projectName: "training" })

User: "Create a new project called Safety Training"
→ First call getClients() to get clientId, then createProject({ name: "Safety Training", clientId: "..." })

## Communication Style

- Be direct and helpful
- Execute tool calls immediately when asked
- Summarize results concisely
- Offer relevant next steps
- If creating a project, always get clients first

## When to Hand Back

Include \`[DONE]\` in your response when:
- You've provided the requested project information
- You've completed the project creation
- The user wants to do something outside project management (e.g., edit nodes, search documents)

Example: "Here are your 5 projects, sorted by last updated. [DONE]"`;

// ============================================================================
// PROJECT AGENT NODE FUNCTION
// ============================================================================

/**
 * The Project Agent node.
 * Handles project management operations.
 */
export async function projectAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[project-agent] ============ Project Agent ============");

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Project management tools
  const projectToolNames = TOOL_CATEGORIES.project;
  
  const projectAgentTools = frontendActions.filter((action: { name: string }) =>
    projectToolNames.includes(action.name)
  );

  console.log("  Available tools:", projectAgentTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total tools:", projectAgentTools.length);

  const systemMessage = new SystemMessage({ content: PROJECT_AGENT_SYSTEM_PROMPT });

  // Bind tools and invoke
  const modelWithTools = projectAgentTools.length > 0
    ? projectAgentModel.bindTools(projectAgentTools)
    : projectAgentModel;

  // Filter messages for this agent's context
  const strippedMessages = stripThinkingBlocks(state.messages || []);
  const slicedMessages = strippedMessages.slice(-10);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[project-agent]");

  console.log("  Invoking project agent model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Project agent response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. If the user asked about projects, call the appropriate tool (listProjects, getProjectDetails, etc.)
2. Provide a helpful response with the project information

The user is waiting for your help with project management.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...recentMessages, nudgeMessage],
      config
    );
    
    aiResponse = response as AIMessage;
    
    if (hasUsableResponse(aiResponse)) {
      console.log("  [RETRY] Success - got usable response on retry");
    }
  }

  return {
    messages: [response],
    currentAgent: "project_agent",
    agentHistory: ["project_agent"],
    routingDecision: null,
  };
}

export default projectAgentNode;


