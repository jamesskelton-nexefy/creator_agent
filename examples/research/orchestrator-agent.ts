/**
 * Orchestrator Agent
 *
 * Coordinates workflow between specialized sub-agents for creating
 * impactful online training content.
 *
 * Sub-agents:
 * - Strategist: Discovers purpose, objectives, scope, constraints
 * - Researcher: Deep knowledge gathering on industry and topics
 * - Architect: Designs structure, presents plan for approval, builds complete skeleton (L2-L6)
 * - Writer: Fills in content within existing Content Block nodes
 * - Visual Designer: Defines aesthetics and writing tone
 *
 * Based on CopilotKit frontend actions pattern:
 * https://docs.copilotkit.ai/langgraph/frontend-actions
 */

import "dotenv/config";
import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { StructuredToolInterface } from "@langchain/core/tools";
import { START, StateGraph, END, Command } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import {
  copilotkitCustomizeConfig,
  copilotkitEmitState,
} from "@copilotkit/sdk-js/langgraph";

// State and agent imports
import {
  OrchestratorStateAnnotation,
  OrchestratorState,
  AgentType,
  ActiveTask,
  summarizeAgentContext,
  generateTaskContext,
} from "./state/agent-state";

// Centralized context management utilities
import {
  processContext,
  MESSAGE_LIMITS,
  TOKEN_LIMITS,
} from "./utils";

import {
  // Creative workflow agents
  strategistNode,
  parseProjectBrief,
  researcherNode,
  researcherTools,
  parseResearchFindings,
  architectNode,
  parseCourseStructure,
  writerNode,
  extractContentOutput,
  visualDesignerNode,
  parseVisualDesign,
  builderAgentNode,
  // Tool-specialized sub-agents
  dataAgentNode,
  projectAgentNode,
  nodeAgentNode,
  documentAgentNode,
  mediaAgentNode,
  frameworkAgentNode,
} from "./agents/index";

// Re-export parsing utilities for external use
export {
  parseProjectBrief,
  parseResearchFindings,
  parseCourseStructure,
  extractContentOutput,
  parseVisualDesign,
};

// Message filtering and trimming now handled by centralized utils/context-management.ts

// ============================================================================
// DEBUG CONFIGURATION
// ============================================================================

/** Enable verbose debug logging via DEBUG_ORCHESTRATOR env var */
const DEBUG = process.env.DEBUG_ORCHESTRATOR === "true";

/** Conditional debug logger - only logs when DEBUG is enabled */
const debugLog = DEBUG 
  ? (...args: any[]) => console.log(...args)
  : () => {};

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const orchestratorModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7,
  // Enable Anthropic prompt caching for system prompts
  // This caches the static system prompt, reducing costs by ~10x for repeated calls
  clientOptions: {
    defaultHeaders: { "anthropic-beta": "prompt-caching-2024-07-31" },
  },
});

// ============================================================================
// TABLE TOOLS EXCLUSION - Tools handled by Data Agent
// ============================================================================

/**
 * Table-specific tools that should be routed to the Data Agent, not handled by supervisor.
 * switchViewMode is kept for basic navigation, but table operations go to data_agent.
 */
const TABLE_TOOLS_FOR_DATA_AGENT = new Set([
  "switchTableViewMode",
  "getTableViewState",
  "getTableFilterOptions",
  "addTableFilter",
  "clearTableFilters",
  "addTableSort",
  "clearTableSorts",
  "setTableGrouping",
  "clearTableGrouping",
  "getTableColumns",
  "showTableColumn",
  "hideTableColumn",
  "exportTableData",
  "listTableViews",
  "saveTableView",
  "loadTableView",
  "deleteTableView",
  "searchTable",
  "getTableDataSummary",
  "getFieldValueDistribution",
]);

// ============================================================================
// TOOL CLASSIFICATION SYSTEM
// ============================================================================

/**
 * Backend tools - executed within LangGraph sub-agent subgraphs.
 * These are NOT handled by supervisor directly - they run in sub-agents.
 */
const BACKEND_TOOL_NAMES = new Set([
  "web_search",
  "tavily_search",
]);

/**
 * Internal routing tools - used by supervisor for agent delegation.
 * These should NEVER be emitted to CopilotKit.
 */
const INTERNAL_ROUTING_TOOLS = new Set([
  "supervisor_response",
]);

/**
 * Gets CopilotKit frontend action names from state.
 */
function getCopilotKitActionNames(state: OrchestratorState): Set<string> {
  const actions = state.copilotkit?.actions ?? [];
  return new Set(actions.map((a: { name: string }) => a.name));
}

/**
 * Tool call classification result.
 */
interface ClassifiedToolCalls {
  backendToolCalls: Array<{ name: string; id?: string; args: any }>;
  frontendToolCalls: Array<{ name: string; id?: string; args: any }>;
  routingToolCalls: Array<{ name: string; id?: string; args: any }>;
}

/**
 * Classifies tool calls by execution target.
 * 
 * Classification priority:
 * 1. Internal routing tools (supervisor_response) -> routingToolCalls
 * 2. Known backend tools (web_search, etc.) -> backendToolCalls
 * 3. CopilotKit frontend actions -> frontendToolCalls
 * 4. Unknown tools -> frontendToolCalls (safer default)
 */
function classifyToolCalls(
  toolCalls: Array<{ name: string; id?: string; args: any }>,
  state: OrchestratorState
): ClassifiedToolCalls {
  const copilotKitActions = getCopilotKitActionNames(state);
  
  const backendToolCalls: ClassifiedToolCalls["backendToolCalls"] = [];
  const frontendToolCalls: ClassifiedToolCalls["frontendToolCalls"] = [];
  const routingToolCalls: ClassifiedToolCalls["routingToolCalls"] = [];
  
  for (const toolCall of toolCalls) {
    if (INTERNAL_ROUTING_TOOLS.has(toolCall.name)) {
      routingToolCalls.push(toolCall);
    } else if (BACKEND_TOOL_NAMES.has(toolCall.name)) {
      backendToolCalls.push(toolCall);
    } else if (copilotKitActions.has(toolCall.name)) {
      frontendToolCalls.push(toolCall);
    } else {
      // Unknown tools default to frontend (safer - visible error vs silent failure)
      console.warn(`  [classifyToolCalls] Unknown tool "${toolCall.name}" - treating as frontend`);
      frontendToolCalls.push(toolCall);
    }
  }
  
  if (toolCalls.length > 0) {
    console.log("  [classifyToolCalls] Results:");
    console.log(`    Routing: ${routingToolCalls.map(t => t.name).join(", ") || "none"}`);
    console.log(`    Backend: ${backendToolCalls.map(t => t.name).join(", ") || "none"}`);
    console.log(`    Frontend: ${frontendToolCalls.map(t => t.name).join(", ") || "none"}`);
  }
  
  return { backendToolCalls, frontendToolCalls, routingToolCalls };
}

// ============================================================================
// SUPERVISOR ROUTING TOOL
// ============================================================================

/**
 * Tool used by the supervisor (orchestrator) to structure responses and route to sub-agents.
 * Following CopilotKit supervisor pattern for multi-agent coordination.
 */
const SUPERVISOR_ROUTING_TOOL = {
  type: "function" as const,
  function: {
    name: "supervisor_response",
    description: "Always use this tool to structure your response and optionally route to a specialized agent.",
    parameters: {
      type: "object",
      properties: {
        answer: {
          type: "string",
          description: "Your response to the user"
        },
        next_agent: {
          type: "string",
          enum: [
            // Creative workflow agents
            "strategist",
            "researcher", 
            "architect",
            "writer",
            "visual_designer",
            "builder_agent",
            // Tool-specialized sub-agents
            "project_agent",
            "node_agent",
            "data_agent",
            "document_agent",
            "media_agent",
            "framework_agent",
            // Complete (no routing)
            "complete"
          ],
          description: "The specialized agent to route to. Use 'complete' if no routing is needed."
        }
      },
      required: ["answer"]
    }
  }
};

// ============================================================================
// ORCHESTRATOR SYSTEM PROMPT
// ============================================================================

const ORCHESTRATOR_SYSTEM_PROMPT = `You are The Orchestrator - the coordinator for a multi-agent system that creates impactful online training content.

## Your Team

You have 7 specialized sub-agents, each with specific capabilities:

### 1. The Strategist
**When to use**: First step for new projects, or when goals/scope are unclear
**Capabilities**: Asks clarifying questions, gathers requirements, defines project brief
**Output**: Project brief with purpose, objectives, scope, constraints

### 2. The Researcher
**When to use**: After strategy is defined, to gather deep knowledge
**Capabilities**: Web search, document search, RAG queries
**Output**: Research findings with industry context, key topics, regulations

### 3. The Architect
**When to use**: After research is complete, to design and build course structure
**Capabilities**: Analyzes hierarchy, designs node structure, presents plan for approval, builds complete skeleton
**Output**: Presents plan to user, gets approval, then creates structure from L2 down to Content Blocks

### 4. The Writer
**When to use**: After Architect has built the structure, to fill in content
**Capabilities**: Updates existing nodes with content, writes text, attaches media
**Output**: Populated Content Block nodes with actual training content

### 5. The Visual Designer
**When to use**: Early in process to define look/feel, or when user asks about design
**Capabilities**: Presents design options, defines colors, fonts, tone
**Output**: Visual design specification for the course

### 6. The Data Agent
**When to use**: For table view operations, data queries, filtering, sorting, grouping
**Capabilities**: Table view filtering, sorting, grouping, column management, data export, view management, project hierarchy awareness
**Output**: Direct table manipulations and data insights

### 7. The Builder Agent
**When to use**: For preview generation, e-learning component rendering, visual preview of content
**Capabilities**: Generates visual previews of content nodes, applies design styles, renders e-learning components (TitleBlock, QuestionBlock, etc.)
**Output**: Generated preview components rendered in Preview view

## Workflow Patterns

### Full Course Creation
1. Strategist → Gather requirements
2. Visual Designer → Define aesthetics (can run parallel with research)
3. Researcher → Deep knowledge gathering
4. Architect → Design structure, present plan, get approval, build complete skeleton (L2-L6)
5. Writer → Fill in content within the existing Content Block nodes

### Quick Content Addition
1. Architect → Plan and build new structure nodes
2. Writer → Fill in the content

### Research Deep-Dive
1. Researcher → Extended research on specific topics

### Design Refresh
1. Visual Designer → New design options

## Your Role

As the Orchestrator, you:
1. **Perform direct actions** - You can do MANY things yourself using frontend tools
2. **Analyze requests** - Understand what the user needs
3. **ALWAYS ask before acting** - Present options and wait for user direction
4. **Manage context** - Pass relevant information between agents
5. **Coordinate workflow** - Ensure proper sequencing

## CRITICAL: Always Ask Before Acting

**This is the most important rule.** After completing any task or receiving information:

1. **Summarize** what was done or learned
2. **Present clear options** for what the user might want to do next
3. **WAIT** for the user to choose before taking action
4. **NEVER** automatically delegate to sub-agents without user approval

**Example response pattern:**
"I've completed [X]. Here are your options:
1. [Option A] - Would you like me to...
2. [Option B] - Or should I...
3. Something else - Tell me what you'd like to do"

**Use the offerOptions tool** to present these choices with clickable buttons.

**When user explicitly requests a specialist**, then and only then route to:
- Say "start research" or "do research" → Researcher
- Say "gather requirements" or "start strategy" → Strategist  
- Say "design structure" or "build course" → Architect
- Say "write content" or "create content" → Writer
- Say "design visuals" or "define aesthetics" → Visual Designer
- Say "table view" or "show data" or "filter nodes" → Data Agent

## Your Tools (Direct Actions)

You have access to ALL frontend tools. Here's how to use each:

**IMPORTANT: Call frontend tools ONE AT A TIME, never in parallel.** When you need information from multiple tools (e.g., getProjectHierarchyInfo AND getAvailableTemplates), call them sequentially - get the result from the first, then call the second. This ensures proper synchronization with the UI.

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
- **showNotification(message, type)** - Show a toast notification (type: "success", "error", "info")

### Edit Mode (REQUIRED before creating/updating nodes)
- **requestEditMode()** - Request edit lock. MUST call this before createNode or updateNodeFields
- **releaseEditMode()** - Release the edit lock when done editing
- **checkEditStatus()** - Check if you have edit access

### Project Management
- **getCurrentProject()** - Get the CURRENT project the user is in. USE THIS FIRST before listProjects when you need to know which project is active!
- **listProjects(searchTerm?, clientId?, sortBy?)** - List ALL projects. sortBy: "updated", "created", "name", "client". Only use when you need to browse/search multiple projects.
- **getProjectDetails(projectId?, projectName?)** - Get project info by ID or name search
- **createProject(name, clientId, description?, templateId?)** - Create new project. Use getClients first for clientId
- **openProjectByName(projectName)** - Search and navigate to a project by name
- **getProjectTemplates()** - List available project templates
- **getClients()** - List available clients for project creation

**IMPORTANT: Project Context Awareness**
The CopilotKit context already tells you which project the user is in. Check the readable context for "currentProject" before calling tools:
- If the user is already in a project, you DON'T need to call listProjects to know which one
- Use getCurrentProject() to confirm the current project if context is unclear
- Only call listProjects when the user explicitly asks to see ALL projects or search for projects

### Node Information (Read-only)
- **getNodeTreeSnapshot(maxNodes?, includeFieldStatus?)** - **RECOMMENDED** Get ALL nodes in ONE call with content status. Eliminates multiple tool calls for structure discovery.
- **getProjectHierarchyInfo()** - Get hierarchy levels, coding config, structure info
- **getNodeChildren(nodeId?)** - Get children of a node (uses selected node if no ID)
- **getNodeDetails(nodeId?)** - Get detailed info about a node
- **getNodesByLevel(level?, levelName?, limit?)** - Find all nodes at a hierarchy level
- **getAvailableTemplates(parentNodeId?)** - Get templates valid for creating under a parent
- **getNodeTemplateFields(templateId)** - Get field schema with assignment IDs for a template
- **getNodeFields(nodeId?)** - Read current field values from a node

### Node Creation & Editing (REQUIRES edit mode)
- **createNode(templateId, title, parentNodeId?, initialFields?)** - Create a node. Use getAvailableTemplates first
- **updateNodeFields(nodeId?, fieldUpdates)** - Update fields. fieldUpdates: { assignmentId: value }

### Template Exploration
- **listAllNodeTemplates(projectTemplateId?, nodeType?, limit?)** - Browse all node templates
- **listFieldTemplates(fieldType?, limit?)** - Browse field templates
- **getTemplateDetails(templateId, templateType?)** - Get full template details. templateType: "node" or "field"

### Document Management
- **uploadDocument(category, instructions?)** - Trigger upload dialog. category: "course_content" or "framework_content"

**IMPORTANT: Document Search is a RESEARCHER-ONLY capability.**
You do NOT have access to document search tools (searchDocuments, listDocuments, searchDocumentsByText, getDocumentLines, getDocumentByName).
These tools belong EXCLUSIVELY to The Researcher agent.

If the user wants to search uploaded documents or analyze document content:
1. Use [ROUTE:researcher] to hand off to The Researcher
2. The Researcher will use the document tools and return findings
3. Do NOT attempt to call these tools yourself - they will fail

You CAN use uploadDocument to help users upload new documents - that's a frontend tool you have access to.

### Media Library (Microverse)
- **searchMicroverse(query?, fileType?, category?, limit?)** - Search media assets
- **getMicroverseDetails(fileId)** - Get detailed info about a media asset
- **getMicroverseUsage(fileId)** - Check where an asset is used
- **attachMicroverseToNode(nodeId, fileId, fieldAssignmentId)** - Attach media to a node field
- **detachMicroverseFromNode(nodeId, fileId, fieldAssignmentId)** - Remove media from a node

### Framework & Criteria Mapping
- **listFrameworks(category?, status?)** - List competency frameworks
- **getFrameworkDetails(frameworkId)** - Get framework info with items
- **searchASQAUnits(query, limit?)** - Search ASQA training units
- **linkFrameworkToProject(frameworkId, projectId)** - Link a framework to project
- **mapCriteriaToNode(nodeId, criteriaId)** - Map criteria to a node
- **suggestCriteriaMappings(nodeId)** - Get AI-suggested criteria mappings

### Table View / Data Operations
**IMPORTANT**: Table view and data queries require the Data Agent specialist.

When user asks for:
- "Show me [something] in table view"
- "Filter/sort/group the table"
- "Find nodes with [condition]"
- "What nodes are under [parent]?" (when table context is implied)
- "Export data"
- Any data exploration or querying task

→ The Data Agent can handle this. Ask the user if they'd like you to involve the Data Agent.

The Data Agent has specialized tools for:
- Switching to and managing table view modes
- Filtering, sorting, and grouping data intelligently
- Understanding field types and operators
- Managing column visibility
- Exporting to CSV
- Saving/loading table view configurations

### User Interaction (USE THESE - don't just type text!)
- **askClarifyingQuestions(questions)** - Ask up to 5 questions with options. Format:
  \`\`\`
  questions: [
    { question: "What is the target audience?", options: ["Beginners", "Intermediate", "Advanced"] },
    { question: "Preferred duration?", options: ["1 hour", "Half day", "Full day"] }
  ]
  \`\`\`
- **offerOptions(title, options, allowMultiple?)** - Present choices to user. Format:
  \`\`\`
  title: "How would you like to proceed?"
  options: [
    { id: "option1", label: "Create project now", description: "Start immediately" },
    { id: "option2", label: "Gather more requirements", description: "Ask more questions first" }
  ]
  \`\`\`
- **requestPlanApproval(plan, title?)** - Get user approval for a plan before executing
- **requestActionApproval(action, reason?)** - Confirm a sensitive action with the user

**IMPORTANT**: When presenting options to the user, ALWAYS use \`offerOptions\` or \`askClarifyingQuestions\` tools.
Do NOT just write out options as text - the tools provide a better UI experience with clickable buttons.

## When to Use Tools Directly vs Delegate to Specialists

### DO IT YOURSELF (use tools directly):
- Creating projects
- Navigating to projects (but NOT "show in table view" requests)
- Listing projects, nodes, templates
- Checking project structure (but NOT table data queries)
- Any quick, straightforward action that doesn't involve table view

**NOTE**: Do NOT create course structure nodes yourself - that's the Architect's job. The Architect designs the plan, presents it to the user for approval, and builds the complete structure.

### INVOLVE SPECIALISTS (only when user explicitly requests):
- **Data Agent**: When user asks for "table view", "show me nodes", data filtering/sorting/grouping
- **Strategist**: When user asks to gather requirements for a new training project
- **Researcher**: When user asks for deep research, web search, document analysis
- **Architect**: When user asks to design/build course structures - the Architect will present its plan and get user approval directly
- **Writer**: When user asks to write/fill in content within existing nodes
- **Visual Designer**: When user asks to define visual aesthetics, colors, fonts, tone

**Remember**: Always ask the user before involving a specialist. Don't auto-route.

**Architect Note**: Once you route to the Architect, it handles the entire structure process including presenting the plan to the user and getting approval. You don't need to intermediate the approval - the Architect uses requestPlanApproval directly.

## Decision Making

When a user message arrives:

1. **Can I do this directly with a tool?** → Use the tool, then offer next steps!
2. **Does this need specialist knowledge?** → Ask user if they want to involve a specialist
3. **Do I need more information?** → Ask using \`askClarifyingQuestions\` tool and WAIT
4. **Is this a greeting or general question?** → Respond and offer options for what to do

## CRITICAL: User-Driven Flow

**ALWAYS wait for user direction.** Never automatically route to sub-agents.

**When user completes an action:**
- Summarize what was done
- Present options using \`offerOptions\` tool
- Wait for user to choose

**USE TOOLS DIRECTLY when asked to:**
- Navigate to projects (NOT table view - ask if they want data_agent)
- List projects, nodes, templates
- Create/update nodes (after getting edit mode)
- Any straightforward operation you can do with tools

**ONLY route to specialists when user explicitly asks:**
- User says "start research" → Then involve Researcher
- User says "gather requirements" → Then involve Strategist
- User says "design the course structure" → Then involve Architect
- User says "write the content" → Then involve Writer
- User says "work on visual design" → Then involve Visual Designer
- User says "show table view" or "filter data" → Then involve Data Agent

## CRITICAL: Handling offerOptions Selections

**When \`offerOptions\` returns "User selected: X", that IS the user's explicit instruction. ACT ON IT IMMEDIATELY.**

The tool result from \`offerOptions\` is how users click buttons in the UI. When you receive:
\`\`\`
"User selected: \"Design the Course Structure\"..."
\`\`\`

This is IDENTICAL to the user typing "design the course structure". You MUST:
1. Acknowledge the selection briefly
2. Route to the appropriate specialist immediately using \`supervisor_response\` with the correct \`next_agent\`

**Mapping offerOptions selections to routing:**
- Selection contains "Design the Course Structure" or "Architect" → \`next_agent: "architect"\`
- Selection contains "Visual Design" or "Designer" → \`next_agent: "visual_designer"\`
- Selection contains "Research" or "Researcher" → \`next_agent: "researcher"\`
- Selection contains "Requirements" or "Strategy" or "Strategist" → \`next_agent: "strategist"\`
- Selection contains "Write" or "Content" or "Writer" → \`next_agent: "writer"\`
- Selection contains "Table" or "Data" → \`next_agent: "data_agent"\`

**DO NOT** respond to an offerOptions selection by:
- Providing another summary
- Offering more options
- Using \`next_agent: "complete"\`

The user already made their choice. Execute it.

## State vs Memory - When to Use Each

**Agent State** (automatic, within-session):
- projectBrief, researchFindings, plannedStructure, courseStructure, visualDesign
- Automatically passed between agents - you don't need to save these
- Populated when agents complete their work and output structured data
- plannedStructure tracks architect's plan + execution progress (survives restarts)

**Memory Tools** (persistent, cross-session):
- saveMemory: ONLY for user preferences or AFTER project completion
- recallMemories: To retrieve past preferences or historical project info

**NEVER use saveMemory for:**
- Project briefs (state handles this)
- Research findings (state handles this)
- Course structures (state handles this)
- Any mid-workflow data

**DO use saveMemory for:**
- User preferences ("I prefer simple language", "I like dark themes")
- Historical project summaries AFTER completion
- Facts about the user that should persist across sessions

## Guidelines

- **ALWAYS use offerOptions or askClarifyingQuestions tools** when presenting choices - never just write options as text
- After ANY action, summarize what was done and offer next steps
- Prefer using tools directly - only involve specialists when user explicitly asks
- When starting new training projects, suggest the Strategist but let user decide
- Keep the user informed of progress
- Summarize agent outputs clearly for the user
- If unsure, ask the user for clarification using the tools (and WAIT for their response)
- **Never auto-route to sub-agents** - always wait for user direction

## Current Context Summary

This section will be updated with the current state of agent outputs.`;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Extracts the most recent substantive user request from messages.
 * Looks for the most recent human message that isn't a simple acknowledgment.
 * Used to set the originalRequest field in activeTask.
 */
function extractOriginalRequest(messages: BaseMessage[]): string {
  // Find the most recent human message that's substantive (not just "ok", "yes", etc.)
  const humanMessages = messages.filter(m => {
    const msgType = (m as any)._getType?.() || (m as any).constructor?.name || "";
    return msgType === "human" || msgType === "HumanMessage";
  });

  // Simple acknowledgments to skip
  const acknowledgments = ["ok", "okay", "yes", "no", "sure", "thanks", "thank you", "got it"];

  // Work backwards to find the most recent substantive message
  for (let i = humanMessages.length - 1; i >= 0; i--) {
    const msg = humanMessages[i];
    const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
    const trimmed = content.trim().toLowerCase();
    
    // Skip if it's just an acknowledgment
    if (acknowledgments.includes(trimmed)) {
      continue;
    }
    
    // Skip very short messages (likely selections or simple responses)
    if (trimmed.length < 10) {
      continue;
    }

    return content;
  }

  // If no substantive message found, use the last human message
  if (humanMessages.length > 0) {
    const lastHuman = humanMessages[humanMessages.length - 1];
    return typeof lastHuman.content === "string" ? lastHuman.content : JSON.stringify(lastHuman.content);
  }

  return "No user request found";
}

/**
 * Generates a goal description for the agent being routed to.
 * Based on the agent type and available context.
 */
function generateGoalForAgent(agent: AgentType, state: OrchestratorState, answer: string): string {
  switch (agent) {
    case "strategist":
      return "Gather project requirements by asking clarifying questions and create a project brief";
    case "researcher":
      return state.projectBrief
        ? `Research and gather information about ${state.projectBrief.industry} for ${state.projectBrief.purpose}`
        : "Conduct research to gather relevant information for the training project";
    case "architect":
      return state.researchFindings
        ? "Design the course structure based on the research findings and project brief"
        : "Design the course structure for the training project";
    case "writer":
      return state.courseStructure
        ? `Create content for the ${state.courseStructure.totalNodes} planned nodes in the course structure`
        : "Write engaging training content following the course structure";
    case "visual_designer":
      return "Define the visual design, color scheme, typography, and writing tone for the training";
    case "builder_agent":
      return "Generate visual previews and render e-learning components";
    case "data_agent":
      return "Handle table view operations, filtering, sorting, and data queries";
    case "document_agent":
      return "Search and analyze uploaded documents";
    case "media_agent":
      return "Search and manage media assets from the library";
    case "framework_agent":
      return "Work with competency frameworks and criteria mapping";
    case "project_agent":
      return "Manage project operations (creation, navigation, listing)";
    case "node_agent":
      return "Handle node operations and template management";
    default:
      return answer; // Use the supervisor's answer as the goal
  }
}

// ============================================================================
// COPILOTKIT HANDLER NODE
// ============================================================================

/**
 * CopilotKit Handler Node
 * 
 * Handles frontend tool calls by:
 * 1. Emitting tool calls to CopilotKit via AG-UI protocol
 * 2. Keeping the graph run OPEN (routing back to supervisor)
 * 3. Allowing the supervisor to receive tool results
 * 
 * CRITICAL: Routes to "supervisor", NOT to END.
 * Routing to END causes orphaned tool calls - the tool result has nowhere to go.
 * 
 * Flow:
 * - Supervisor detects frontend tool calls
 * - Routes to copilotkit_handler with pendingFrontendActions
 * - Handler emits tool calls to CopilotKit
 * - Routes back to supervisor
 * - Supervisor's safety check waits for tool results
 * - Tool results flow back from CopilotKit
 * - Supervisor processes results and responds
 */
async function copilotKitHandlerNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Command> {
  console.log("\n========================================");
  console.log("[copilotkit_handler] Processing frontend tool calls");
  console.log("========================================");
  
  try {
    const pendingActions = state.pendingFrontendActions ?? [];
    console.log(`  Pending frontend actions: ${pendingActions.length}`);
    
    if (pendingActions.length > 0) {
      console.log(`  Actions: ${pendingActions.map(a => a.name).join(", ")}`);
    }
    
    // Configure CopilotKit to emit tool calls
    const modifiedConfig = copilotkitCustomizeConfig(config, {
      emitToolCalls: true,
      emitMessages: true,
    });
    
    // Emit current state to CopilotKit
    // This triggers the frontend to execute the pending tool calls
    await copilotkitEmitState(modifiedConfig, state);
    
    console.log("  Tool calls emitted to CopilotKit");
    console.log("  -> Routing to END to await tool results from CopilotKit");
    
    // CRITICAL: Route to END after emitting tool calls
    // The graph run finishes, but CopilotKit has received the tool call info.
    // CopilotKit will execute the frontend tool and send back a ToolMessage,
    // which starts a new run where supervisor can process the result.
    return new Command({
      goto: END,
      update: {
        pendingFrontendActions: [],
        currentAgent: "orchestrator" as AgentType,
      },
    });
    
  } catch (error) {
    console.error("  [copilotkit_handler] Error:", error);
    
    // On error, still route to supervisor with error state
    return new Command({
      goto: "supervisor",
      update: {
        pendingFrontendActions: [],
        currentAgent: "orchestrator" as AgentType,
        lastError: error instanceof Error ? error.message : String(error),
      },
    });
  }
}

// ============================================================================
// SUPERVISOR (ORCHESTRATOR) NODE - CopilotKit Supervisor Pattern
// ============================================================================

/**
 * Supervisor agent that coordinates specialized sub-agents.
 * Uses Command pattern for routing instead of conditional edges.
 * 
 * Follows CopilotKit multi-agent supervisor pattern:
 * - Binds SUPERVISOR_ROUTING_TOOL for structured routing decisions
 * - Returns Command({ goto: next_agent }) for routing
 * - Returns Command({ goto: END }) when waiting for user/tool results
 */
async function supervisorNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Command> {
  console.log("\n[supervisor] ============ Supervisor Agent ============");
  console.log("  Agent context:", summarizeAgentContext(state));
  console.log("  Messages count:", state.messages?.length ?? 0);
  console.log("  Current agent:", state.currentAgent);
  console.log("  Agent history:", state.agentHistory?.join(" -> ") || "none");

  // SAFETY CHECK: Detect if the last message is an AIMessage with pending tool_calls
  // If so, we're waiting for CopilotKit to send the tool_result - don't invoke LLM
  const messages = state.messages || [];
  if (messages.length > 0) {
    const lastMsg = messages[messages.length - 1];
    const lastMsgType = (lastMsg as any)._getType?.() || (lastMsg as any).constructor?.name || "";
    
    if (lastMsgType === "ai" || lastMsgType === "AIMessage" || lastMsgType === "AIMessageChunk") {
      const aiMsg = lastMsg as AIMessage;
      if (aiMsg.tool_calls?.length) {
        // Check if any tool_calls are unresolved (no matching tool_result in messages)
        const toolResultIds = new Set(
          messages
            .filter(m => {
              const t = (m as any)._getType?.() || (m as any).constructor?.name || "";
              return t === "tool" || t === "ToolMessage";
            })
            .map(m => (m as ToolMessage).tool_call_id)
        );
        
        const unresolvedCalls = aiMsg.tool_calls.filter(tc => tc.id && !toolResultIds.has(tc.id));
        
        if (unresolvedCalls.length > 0) {
          console.log(`  [SAFETY] ${unresolvedCalls.length} unresolved tool_calls: ${unresolvedCalls.map(tc => tc.name).join(", ")}`);
          
          // Classify unresolved calls for logging
          const { frontendToolCalls } = classifyToolCalls(unresolvedCalls, state);
          
          if (frontendToolCalls.length > 0) {
            console.log("  [SAFETY] Unresolved frontend tools - waiting for CopilotKit response");
          }
          
          // Route to END to wait for tool results from CopilotKit
          // If these are frontend tools, they've already been emitted by copilotkit_handler
          // in a previous run. CopilotKit will send the ToolMessage to start a new run.
          console.log("  [SAFETY] Routing to END - waiting for tool results");
          return new Command({
            goto: END,
            update: { currentAgent: "orchestrator" as AgentType },
          });
        }
      }
    }
  }

  // CHECK: If a sub-agent has active work in progress, route back to them
  // This handles phase-based workflows where agents need to continue their work
  // The agent is responsible for clearing awaitingUserAction when work is complete
  if (state.awaitingUserAction) {
    const workState = state.awaitingUserAction;
    const awaitingAgent = workState.agent as AgentType;
    console.log(`  [WORK STATE] Agent "${awaitingAgent}" has active work in progress`);
    console.log(`  [WORK STATE] Phase: ${workState.phase}`);
    console.log(`  [WORK STATE] Pending tool: ${workState.pendingTool || "none"}`);
    console.log(`  [WORK STATE] Allowed tools: ${workState.allowedTools?.join(", ") || "none"}`);
    console.log(`  [WORK STATE] Routing back to ${awaitingAgent} to continue work`);
    return new Command({
      goto: awaitingAgent,
      update: { 
        // DON'T clear awaitingUserAction - let the agent manage its own state
        currentAgent: awaitingAgent,
      },
    });
  }

  // Get frontend tools - supervisor has most tools but NOT table-specific ones
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Supervisor has access to CopilotKit frontend tools EXCEPT table-specific ones
  // Table operations are routed to the Data Agent for specialized handling
  const frontendTools = frontendActions.filter(
    (action: { name: string }) => !TABLE_TOOLS_FOR_DATA_AGENT.has(action.name)
  );

  console.log("  Available CopilotKit tools:", frontendTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total frontend tools:", frontendTools.length);

  // Build dynamic system prompt with context
  let systemContent = ORCHESTRATOR_SYSTEM_PROMPT;

  // Add active task context (most important - what we're currently working on)
  if (state.activeTask) {
    systemContent += `\n\n## ACTIVE TASK (CRITICAL - This is what we're working on)

**Original User Request:**
${state.activeTask.originalRequest}

**Current Goal:** ${state.activeTask.currentGoal}
**Assigned to:** ${state.activeTask.assignedAgent}
**Started:** ${state.activeTask.startedAt}
${state.activeTask.progress.length > 0 ? `\n**Progress Made:**\n${state.activeTask.progress.map(p => `- ${p}`).join("\n")}` : ""}

IMPORTANT: Keep this task in mind when routing to agents. If the task is not complete, 
continue routing to the appropriate agent to finish it.`;
  }

  // Add current context summary
  systemContent += `\n\n## Current Agent Context\n`;

  if (state.projectBrief) {
    systemContent += `\n### Project Brief (from Strategist) ✓
- Purpose: ${state.projectBrief.purpose}
- Industry: ${state.projectBrief.industry}
- Target Audience: ${state.projectBrief.targetAudience}
- Objectives: ${state.projectBrief.objectives.length} defined`;
  } else {
    systemContent += `\n### Project Brief: Not yet defined
Consider starting with the Strategist to gather requirements.`;
  }

  if (state.researchFindings) {
    systemContent += `\n\n### Research Findings (from Researcher) ✓
- Key Topics: ${state.researchFindings.keyTopics.length}
- Regulations: ${state.researchFindings.regulations.length}
- Citations: ${state.researchFindings.citations.length}`;
  } else {
    systemContent += `\n\n### Research: Not yet conducted`;
  }

  if (state.plannedStructure) {
    const executed = Object.keys(state.plannedStructure.executedNodes || {}).length;
    const total = state.plannedStructure.nodes.length;
    const status = state.plannedStructure.executionStatus;
    
    systemContent += `\n\n### Course Plan (from Architect) ${status === 'completed' ? '✓' : '⏳'}
- Status: ${status.toUpperCase()}
- Progress: ${executed}/${total} nodes created
- Plan saved at: ${state.plannedStructure.plannedAt}`;
    
    if (status === 'in_progress' || status === 'planned') {
      systemContent += `\n- **Action needed**: Route to architect to continue building structure`;
    }
  }

  if (state.courseStructure) {
    systemContent += `\n\n### Course Structure (from Architect) ✓
- Total Nodes: ${state.courseStructure.totalNodes}
- Max Depth: ${state.courseStructure.maxDepth}
- Summary: ${state.courseStructure.summary}`;
  } else if (!state.plannedStructure) {
    systemContent += `\n\n### Course Structure: Not yet designed`;
  }

  if (state.visualDesign) {
    systemContent += `\n\n### Visual Design (from Visual Designer) ✓
- Theme: ${state.visualDesign.theme}
- Tone: ${state.visualDesign.writingTone.tone}
- Style: ${state.visualDesign.typography.style}`;
  } else {
    systemContent += `\n\n### Visual Design: Not yet defined`;
  }

  if (state.writtenContent && state.writtenContent.length > 0) {
    systemContent += `\n\n### Written Content (from Writer) ✓
- Nodes Created: ${state.writtenContent.length}`;
  }

  // Add routing instructions for supervisor
  systemContent += `\n\n## Routing Instructions

**Use the supervisor_response tool** to structure your response and route to sub-agents.

### Available sub-agents:
- **strategist** - For requirements gathering, project brief creation
- **researcher** - For knowledge gathering, web search, document search
- **architect** - For course structure design, plan presentation, approval, and building (handles entire structure workflow)
- **writer** - For filling in content within existing Content Block nodes
- **visual_designer** - For design and aesthetics
- **builder_agent** - For preview generation, e-learning component rendering
- **project_agent** - For project listing, creation, navigation
- **node_agent** - For node operations, template management, edit mode
- **data_agent** - For table view operations, filtering, sorting, grouping
- **document_agent** - For document search (RAG), document management
- **media_agent** - For media library (Microverse) operations
- **framework_agent** - For competency framework and criteria mapping
- **complete** - When no routing is needed (asking user questions, waiting for input)

### When to route:
- User explicitly requests a specialist
- Task requires specialized tools the supervisor doesn't have
- Complex operation that benefits from specialized context

### When NOT to route (use 'complete'):
- Asking user a question
- Performing actions with your own tools
- User interaction and clarification`;

  // Create system message with prompt caching
  const systemMessage = new SystemMessage({
    content: [
      {
        type: "text",
        text: ORCHESTRATOR_SYSTEM_PROMPT,
        cache_control: { type: "ephemeral" },
      },
      {
        type: "text",
        text: systemContent.replace(ORCHESTRATOR_SYSTEM_PROMPT, ""),
      },
    ],
  });

  // Bind SUPERVISOR_ROUTING_TOOL along with CopilotKit frontend tools
  // The supervisor can call frontend tools directly OR route to sub-agents
  const allTools = [...frontendTools, SUPERVISOR_ROUTING_TOOL];
  const modelWithTools = orchestratorModel.bindTools(allTools);

  // Prepare messages with unified context management pipeline
  // Balanced context reduction - retain node/project info to prevent re-fetching loops
  // NOTE: Skip summarization if we already have a summary - don't regenerate unnecessarily
  const hasExistingSummary = !!state.conversationSummary;
  if (hasExistingSummary) {
    console.log("  [SUMMARY] Using existing summary from state (skipping regeneration)");
  }
  
  const contextResult = await processContext(state.messages || [], {
    maxTokens: TOKEN_LIMITS.orchestrator,
    fallbackMessageCount: MESSAGE_LIMITS.orchestrator,
    // CRITICAL: Strip large jsxCode from older generateCustomComponent calls
    // This prevents state explosion from builder agent's component generation
    enableToolArgStripping: true,
    toolArgStripKeepCount: 3,       // Keep 3 most recent with full JSX
    // Compression: Reduce verbose tool results but keep more recent ones
    enableToolCompression: true,
    compressionKeepCount: 5,        // Increased from 2 - keep more full results
    compressionMaxLength: 1000,     // Increased from 500 - higher threshold
    // Clearing: Replace old tool results but preserve critical ones
    enableToolClearing: true,
    toolKeepCount: 10,              // Increased from 3 - retain more history
    excludeTools: [                 // Never clear these critical tools
      'getNodeFields',
      'getNodeDetails',
      'getNodeChildren',
      'getCurrentProject',
      'getProjectHierarchyInfo',
      'batchCreateNodes',            // Critical for tracking created nodes
      'createNode',                  // Track individual node creation
    ],
    // Summarization: Only if we don't have an existing summary
    // Once summarized, the summary persists in state and we don't regenerate
    enableSummarization: !hasExistingSummary,
    summarizeTriggerTokens: 100000, // Trigger at 100k tokens (very long conversations)
    summarizeKeepMessages: 15,      // Keep 15 most recent messages after summarizing
    logPrefix: "[supervisor]",
    // Task preservation: Keep messages containing the original user request
    originalRequest: state.activeTask?.originalRequest,
    // CRITICAL: Include batchCreateNodes keywords to prevent re-execution after context trimming
    preserveKeywords: [
      "project brief", "training", "course", "module", "lesson",
      "batchCreateNodes", "tempIdToNodeId", "nodesCreatedCount", "completedTempIds",
      "Successfully created", "DO NOT call batchCreateNodes again",
    ],
  });

  const filteredMessages = contextResult.messages;
  // Use newly generated summary OR existing summary from state
  const conversationSummary = contextResult.summary ?? state.conversationSummary;

  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
  });

  console.log("  Invoking supervisor model...");

  // Build system message - include summary if we have one
  // Anthropic only allows ONE SystemMessage at position 0, so we combine them
  let finalSystemMessage: SystemMessage;
  if (conversationSummary) {
    console.log("  [SUMMARY] Including conversation summary in system prompt");
    finalSystemMessage = new SystemMessage({
      content: [
        {
          type: "text",
          text: `[Previous Conversation Summary]\n${conversationSummary}\n\n---\n\n`,
        },
        {
          type: "text",
          text: ORCHESTRATOR_SYSTEM_PROMPT,
          cache_control: { type: "ephemeral" },
        },
        {
          type: "text",
          text: systemContent.replace(ORCHESTRATOR_SYSTEM_PROMPT, ""),
        },
      ],
    });
  } else {
    finalSystemMessage = systemMessage;
  }

  const response = await modelWithTools.invoke(
    [finalSystemMessage, ...filteredMessages],
    customConfig
  );

  console.log("  Supervisor response received");

  const aiResponse = response as AIMessage;
  // CRITICAL: Use filteredMessages (trimmed/summarized) as base, NOT full state.messages
  // This ensures the persisted state stays bounded and doesn't cause
  // "RangeError: Invalid string length" during JSON serialization by FileSystemPersistence
  let updatedMessages = [...filteredMessages, response];

  // Handle tool calls for routing
  if (aiResponse.tool_calls && aiResponse.tool_calls.length > 0) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
    
    // Check if supervisor_response tool was called for routing
    const routingToolCall = aiResponse.tool_calls.find(tc => tc.name === "supervisor_response");
    
    if (routingToolCall) {
      const args = routingToolCall.args as { answer: string; next_agent?: string };
      const nextAgent = args.next_agent;
      
      // Add tool response message
      const toolResponse = new ToolMessage({
        tool_call_id: routingToolCall.id!,
        content: `Routing to ${nextAgent || 'complete'} and providing the answer`,
      });
      
      // Add the answer as an AIMessage
      const answerMessage = new AIMessage({ content: args.answer });
      
      updatedMessages = [...updatedMessages, toolResponse, answerMessage];
      
      // Route to sub-agent if specified and not 'complete'
      if (nextAgent && nextAgent !== "complete") {
        // State validation warnings for agents that expect prior work
        if (nextAgent === "architect" && !state.projectBrief) {
          console.warn("[supervisor] WARNING: Routing to architect but projectBrief is null. Consider running strategist first.");
        }
        if (nextAgent === "architect" && !state.researchFindings) {
          console.warn("[supervisor] WARNING: Routing to architect but researchFindings is null. Consider running researcher first.");
        }
        if (nextAgent === "writer" && !state.courseStructure) {
          console.warn("[supervisor] WARNING: Routing to writer but courseStructure is null. Consider running architect first.");
        }
        
        // Create or update activeTask for context persistence
        // This ensures agents know what they're working on even after context trimming
        const originalRequest = state.activeTask?.originalRequest || extractOriginalRequest(updatedMessages);
        const currentGoal = generateGoalForAgent(nextAgent as AgentType, state, args.answer);
        
        const newActiveTask: ActiveTask = {
          originalRequest,
          currentGoal,
          assignedAgent: nextAgent as AgentType,
          progress: state.activeTask?.progress || [],
          startedAt: state.activeTask?.startedAt || new Date().toISOString(),
          agentInstructions: args.answer, // Use the supervisor's answer as instructions
        };
        
        console.log(`  -> Routing to sub-agent: ${nextAgent}`);
        console.log(`  -> Active task: ${currentGoal.substring(0, 80)}...`);
        
        // Clear writer state when LEAVING the writer agent to go to a different agent
        // This ensures fresh context on next writer invocation for a new task
        const writerStateClear: Partial<OrchestratorState> = {};
        if (state.currentAgent === "writer" && nextAgent !== "writer") {
          console.log("  -> Clearing writer state (leaving writer agent)");
          writerStateClear.writerMessages = [];
          writerStateClear.writerProgress = null;
        }
        
        return new Command({
          goto: nextAgent,
          update: {
            messages: updatedMessages,
            currentAgent: nextAgent as AgentType,
            agentHistory: [nextAgent as AgentType],
            activeTask: newActiveTask,
            conversationSummary, // Persist summary in state
            ...writerStateClear,
          },
        });
      }
      
      // supervisor_response with 'complete' or no next_agent - return with answer
      console.log("  -> Routing to END (supervisor response complete)");
      
      // Clear writer state when completing a task (if writer was the last active agent)
      const writerStateClearOnComplete: Partial<OrchestratorState> = {};
      if (state.currentAgent === "writer") {
        console.log("  -> Clearing writer state (task complete from writer)");
        writerStateClearOnComplete.writerMessages = [];
        writerStateClearOnComplete.writerProgress = null;
      }
      
      return new Command({
        goto: END,
        update: {
          messages: updatedMessages,
          currentAgent: "orchestrator" as AgentType,
          conversationSummary, // Persist summary in state
          ...writerStateClearOnComplete,
        },
      });
    }
    
    // Classify tool calls for logging
    const { frontendToolCalls } = classifyToolCalls(
      aiResponse.tool_calls,
      state
    );

    if (frontendToolCalls.length > 0) {
      console.log(`  -> Frontend tools: ${frontendToolCalls.map(t => t.name).join(", ")}`);
    }

    // Route to END for frontend tool calls
    // Tool calls are already emitted via copilotkitCustomizeConfig during LLM invocation
    // CopilotKit will execute the frontend tool and send back a ToolMessage
    // which starts a new run where supervisor processes the result
    console.log("  -> Routing to END (waiting for tool results from CopilotKit)");
    return new Command({
      goto: END,
      update: {
        messages: updatedMessages,
        currentAgent: "orchestrator" as AgentType,
        conversationSummary, // Persist summary in state
      },
    });
  }

  // No tool calls - end and wait for user input
  console.log("  -> Routing to END (waiting for user input)");
  return new Command({
    goto: END,
    update: {
      messages: updatedMessages,
      currentAgent: "orchestrator" as AgentType,
    },
  });
}

// ============================================================================
// SUBGRAPH CREATION - CopilotKit Supervisor Pattern
// ============================================================================

console.log("[supervisor] Creating sub-agent subgraphs...");

/**
 * Creates a subgraph wrapper for a sub-agent node function.
 * Following CopilotKit multi-agent supervisor pattern:
 * - Each sub-agent is a compiled StateGraph
 * - Added as a node to the main workflow
 * - Simple edge back to supervisor after completion
 */
function createSubAgentGraph(
  nodeFunction: (state: OrchestratorState, config: RunnableConfig) => Promise<Partial<OrchestratorState>>
) {
  // Use type assertion for StateGraph builder pattern
  // The node name doesn't matter since this is a single-node subgraph
  return new StateGraph(OrchestratorStateAnnotation)
    .addNode("agent" as any, nodeFunction)
    .addEdge(START, "agent" as any)
    .addEdge("agent" as any, END)
    .compile();
}

/**
 * Creates a subgraph with tool execution capability.
 * Used for agents that have backend tools (e.g., researcher with web_search).
 * 
 * The subgraph loops: agent -> tools -> agent until no more tool calls.
 * This allows the agent to call tools multiple times before returning to supervisor.
 */
function createSubAgentGraphWithTools(
  nodeFunction: (state: OrchestratorState, config: RunnableConfig) => Promise<Partial<OrchestratorState>>,
  tools: StructuredToolInterface[]
) {
  const toolNode = new ToolNode(tools);
  
  // Route based on whether the last message has tool calls
  function shouldContinueToTools(state: OrchestratorState): "tools" | typeof END {
    const messages = state.messages || [];
    if (messages.length === 0) return END;
    
    const lastMsg = messages[messages.length - 1];
    const msgType = (lastMsg as any)._getType?.() || (lastMsg as any).constructor?.name || "";
    
    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = lastMsg as AIMessage;
      // Check if there are tool calls that match our backend tools
      if (aiMsg.tool_calls?.length) {
        const toolNames = new Set(tools.map(t => t.name));
        const hasBackendToolCalls = aiMsg.tool_calls.some(tc => toolNames.has(tc.name));
        if (hasBackendToolCalls) {
          console.log(`  [subgraph] Routing to tools: ${aiMsg.tool_calls.map(tc => tc.name).join(", ")}`);
          return "tools";
        }
      }
    }
    return END;
  }
  
  return new StateGraph(OrchestratorStateAnnotation)
    .addNode("agent" as any, nodeFunction)
    .addNode("tools" as any, toolNode)
    .addEdge(START, "agent" as any)
    .addConditionalEdges("agent" as any, shouldContinueToTools)
    .addEdge("tools" as any, "agent" as any)
    .compile();
}

// Create subgraphs for each sub-agent
// Creative workflow agents
const strategistSubgraph = createSubAgentGraph(strategistNode);
// Researcher has backend tools (web_search, document search) - use tool-enabled subgraph
const researcherSubgraph = createSubAgentGraphWithTools(researcherNode, researcherTools);
const architectSubgraph = createSubAgentGraph(architectNode);
const writerSubgraph = createSubAgentGraph(writerNode);
const visualDesignerSubgraph = createSubAgentGraph(visualDesignerNode);
const builderAgentSubgraph = createSubAgentGraph(builderAgentNode);

// Tool-specialized sub-agents
const projectAgentSubgraph = createSubAgentGraph(projectAgentNode);
const nodeAgentSubgraph = createSubAgentGraph(nodeAgentNode);
const dataAgentSubgraph = createSubAgentGraph(dataAgentNode);
const documentAgentSubgraph = createSubAgentGraph(documentAgentNode);
const mediaAgentSubgraph = createSubAgentGraph(mediaAgentNode);
const frameworkAgentSubgraph = createSubAgentGraph(frameworkAgentNode);

console.log("[supervisor] Sub-agent subgraphs created successfully");

// ============================================================================
// GRAPH DEFINITION - Supervisor Pattern with Subgraphs
// ============================================================================

console.log("[supervisor] Building main workflow graph...");

// Define all possible routing destinations for the supervisor
const SUPERVISOR_ROUTING_DESTINATIONS = [
  // Creative workflow agents
  "strategist",
  "researcher",
  "architect", 
  "writer",
  "visual_designer",
  "builder_agent",
  // Tool-specialized sub-agents
  "project_agent",
  "node_agent",
  "data_agent",
  "document_agent",
  "media_agent",
  "framework_agent",
  // CopilotKit handler for frontend tool calls
  "copilotkit_handler",
  // End (wait for user/CopilotKit)
  END,
];

const workflow = new StateGraph(OrchestratorStateAnnotation)
  // Supervisor node - uses Command for routing
  .addNode("supervisor", supervisorNode, { 
    ends: SUPERVISOR_ROUTING_DESTINATIONS 
  })
  
  // CopilotKit handler for frontend tool calls
  // Routes back to supervisor after emitting tool calls to keep run open
  .addNode("copilotkit_handler", copilotKitHandlerNode)
  
  // Creative workflow subgraphs
  .addNode("strategist", strategistSubgraph)
  .addNode("researcher", researcherSubgraph)
  .addNode("architect", architectSubgraph)
  .addNode("writer", writerSubgraph)
  .addNode("visual_designer", visualDesignerSubgraph)
  .addNode("builder_agent", builderAgentSubgraph)
  
  // Tool-specialized sub-agent subgraphs
  .addNode("project_agent", projectAgentSubgraph)
  .addNode("node_agent", nodeAgentSubgraph)
  .addNode("data_agent", dataAgentSubgraph)
  .addNode("document_agent", documentAgentSubgraph)
  .addNode("media_agent", mediaAgentSubgraph)
  .addNode("framework_agent", frameworkAgentSubgraph)

  // Entry point -> supervisor
  .addEdge(START, "supervisor")
  
  // CopilotKit handler routes back to supervisor to await tool results
  .addEdge("copilotkit_handler", "supervisor")

  // Simple edges: sub-agents route back to supervisor after completion
  // Following CopilotKit supervisor pattern - no conditional edges needed
  .addEdge("strategist", "supervisor")
  .addEdge("researcher", "supervisor")
  .addEdge("architect", "supervisor")
  .addEdge("writer", "supervisor")
  .addEdge("visual_designer", "supervisor")
  .addEdge("builder_agent", "supervisor")
  .addEdge("project_agent", "supervisor")
  .addEdge("node_agent", "supervisor")
  .addEdge("data_agent", "supervisor")
  .addEdge("document_agent", "supervisor")
  .addEdge("media_agent", "supervisor")
  .addEdge("framework_agent", "supervisor");

// ============================================================================
// DEBUG UTILITIES
// ============================================================================

/**
 * Debug utility: Logs the current state of tool calls in message history.
 * Useful for debugging orphaned tool call issues.
 */
function debugToolCallState(messages: BaseMessage[], prefix: string = ""): void {
  const toolCalls = new Map<string, { name: string; hasResult: boolean }>();
  
  for (const msg of messages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    
    if (msgType === "ai" || msgType === "AIMessage") {
      for (const tc of (msg as AIMessage).tool_calls || []) {
        if (tc.id) toolCalls.set(tc.id, { name: tc.name, hasResult: false });
      }
    }
    
    if (msgType === "tool" || msgType === "ToolMessage") {
      const existing = toolCalls.get((msg as ToolMessage).tool_call_id);
      if (existing) existing.hasResult = true;
    }
  }
  
  console.log(`${prefix}=== TOOL CALL STATE ===`);
  if (toolCalls.size === 0) {
    console.log(`${prefix}  No tool calls in history`);
    return;
  }
  
  for (const [id, state] of toolCalls) {
    const status = state.hasResult ? "resolved" : "PENDING";
    console.log(`${prefix}  ${state.name} (${id.slice(0, 8)}...): ${status}`);
  }
  
  const pendingCount = [...toolCalls.values()].filter(tc => !tc.hasResult).length;
  if (pendingCount > 0) {
    console.log(`${prefix}  WARNING: ${pendingCount} pending tool call(s)!`);
  }
}

/**
 * Finds orphaned tool calls (calls without matching results).
 * Returns array of orphaned tool call IDs.
 */
function findOrphanedToolCalls(messages: BaseMessage[]): string[] {
  const toolCallIds = new Set<string>();
  const toolResultIds = new Set<string>();
  
  for (const msg of messages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    
    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      for (const tc of (msg as AIMessage).tool_calls || []) {
        if (tc.id) toolCallIds.add(tc.id);
      }
    }
    
    if (msgType === "tool" || msgType === "ToolMessage") {
      toolResultIds.add((msg as ToolMessage).tool_call_id);
    }
  }
  
  // Find tool calls without results
  return [...toolCallIds].filter(id => !toolResultIds.has(id));
}

// Export debug utilities for use in testing
export { debugToolCallState, findOrphanedToolCalls };

// ============================================================================
// PERSISTENCE SETUP
// ============================================================================

// In LangSmith Cloud, checkpointer is provided automatically
// Only use custom PostgresSaver for local development
const SUPABASE_DB_URL = process.env.SUPABASE_DB_URL;
const IS_LANGSMITH_CLOUD = !SUPABASE_DB_URL;

let checkpointer: PostgresSaver | undefined;

if (!IS_LANGSMITH_CLOUD && SUPABASE_DB_URL) {
  console.log("[supervisor] Initializing PostgreSQL checkpointer...");
  console.log("[supervisor] DB URL:", SUPABASE_DB_URL.replace(/:[^:@]+@/, ":****@"));
  
  checkpointer = PostgresSaver.fromConnString(SUPABASE_DB_URL);
  await checkpointer.setup();
  console.log("[supervisor] PostgreSQL checkpointer initialized successfully");
} else {
  console.log("[supervisor] Running in LangSmith Cloud - using managed checkpointer");
}

// Compile the graph
export const agent = workflow.compile({
  ...(checkpointer && { checkpointer }),
});

console.log("[supervisor] Workflow graph compiled successfully");
console.log("[supervisor] Architecture: CopilotKit Supervisor Pattern with Subgraphs");
console.log("[supervisor] Nodes: supervisor + 11 sub-agent subgraphs + backend_tools");
console.log("[supervisor] Sub-agents: strategist, researcher, architect, writer, visual_designer,");
console.log("[supervisor]            project_agent, node_agent, data_agent, document_agent, media_agent, framework_agent");


