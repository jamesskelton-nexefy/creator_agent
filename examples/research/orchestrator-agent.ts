/**
 * Orchestrator Agent
 *
 * Coordinates workflow between specialized sub-agents for creating
 * impactful online training content.
 *
 * Sub-agents:
 * - Strategist: Discovers purpose, objectives, scope, constraints
 * - Researcher: Deep knowledge gathering on industry and topics
 * - Architect: Structures course for maximum learning impact
 * - Writer: Creates Level 6 content nodes
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
import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";

// State and agent imports
import {
  OrchestratorStateAnnotation,
  OrchestratorState,
  AgentType,
  summarizeAgentContext,
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
**When to use**: After research is complete, to design course structure
**Capabilities**: Analyzes hierarchy, designs node structure, plans content flow
**Output**: Course structure with planned nodes and learning progression

### 4. The Writer
**When to use**: After structure is approved, to create actual content
**Capabilities**: Creates nodes, writes content, manages edit mode
**Output**: Written content nodes following the structure

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
4. Architect → Design course structure
5. Get approval → User reviews structure
6. Writer → Create content nodes

### Quick Content Addition
1. Architect → Plan new content location
2. Writer → Create the content

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
- Creating simple nodes (structural, categories)
- Checking project structure (but NOT table data queries)
- Any quick, straightforward action that doesn't involve table view

### INVOLVE SPECIALISTS (only when user explicitly requests):
- **Data Agent**: When user asks for "table view", "show me nodes", data filtering/sorting/grouping
- **Strategist**: When user asks to gather requirements for a new training project
- **Researcher**: When user asks for deep research, web search, document analysis
- **Architect**: When user asks to design course structures with learning progressions
- **Writer**: When user asks to create rich content nodes with detailed content
- **Visual Designer**: When user asks to define visual aesthetics, colors, fonts, tone

**Remember**: Always ask the user before involving a specialist. Don't auto-route.

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
          console.log(`  [SAFETY] Last message has ${unresolvedCalls.length} unresolved tool_calls: ${unresolvedCalls.map(tc => tc.name).join(", ")}`);
          console.log("  [SAFETY] Waiting for tool_result from CopilotKit - skipping LLM invocation");
          // Return Command to END - wait for CopilotKit to send tool_result
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
- **architect** - For course structure design, hierarchy planning
- **writer** - For content creation, node writing
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
  const filteredMessages = await processContext(state.messages || [], {
    maxTokens: TOKEN_LIMITS.orchestrator,
    fallbackMessageCount: MESSAGE_LIMITS.orchestrator,
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
      'getProjectHierarchyInfo'
    ],
    // Summarization: Less aggressive condensation
    enableSummarization: true,
    summarizeTriggerTokens: 25000,  // Increased from 15000 - trigger later
    summarizeKeepMessages: 15,      // Increased from 8 - keep more messages
    logPrefix: "[supervisor]",
  });

  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
  });

  console.log("  Invoking supervisor model...");

  const response = await modelWithTools.invoke(
    [systemMessage, ...filteredMessages],
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
        
        console.log(`  -> Routing to sub-agent: ${nextAgent}`);
        return new Command({
          goto: nextAgent,
          update: {
            messages: updatedMessages,
            currentAgent: nextAgent as AgentType,
            agentHistory: [nextAgent as AgentType],
          },
        });
      }
      
      // supervisor_response with 'complete' or no next_agent - return with answer
      console.log("  -> Routing to END (supervisor response complete)");
      return new Command({
        goto: END,
        update: {
          messages: updatedMessages,
          currentAgent: "orchestrator" as AgentType,
        },
      });
    }
    
    // CopilotKit frontend tool call (not supervisor_response) - route to END for CopilotKit to execute
    console.log("  -> Routing to END (CopilotKit tool execution)");
    return new Command({
      goto: END,
      update: {
        messages: updatedMessages,
        currentAgent: "orchestrator" as AgentType,
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
  // End (wait for user/CopilotKit)
  END,
];

const workflow = new StateGraph(OrchestratorStateAnnotation)
  // Supervisor node - uses Command for routing
  .addNode("supervisor", supervisorNode, { 
    ends: SUPERVISOR_ROUTING_DESTINATIONS 
  })
  
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


