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
import { START, StateGraph, END } from "@langchain/langgraph";
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
  filterOrphanedToolResults,
  trimMessages,
  summarizeIfNeeded,
  MESSAGE_LIMITS,
  TOKEN_LIMITS,
} from "./utils";

import {
  strategistNode,
  parseProjectBrief as _parseProjectBrief,
  researcherNode,
  researcherTools,
  parseResearchFindings as _parseResearchFindings,
  architectNode,
  parseCourseStructure as _parseCourseStructure,
  writerNode,
  extractContentOutput as _extractContentOutput,
  visualDesignerNode,
  parseVisualDesign as _parseVisualDesign,
  dataAgentNode,
} from "./agents/index";

// Re-export parsing utilities for external use
export {
  _parseProjectBrief as parseProjectBrief,
  _parseResearchFindings as parseResearchFindings,
  _parseCourseStructure as parseCourseStructure,
  _extractContentOutput as extractContentOutput,
  _parseVisualDesign as parseVisualDesign,
};

// Message filtering and trimming now handled by centralized utils/context-management.ts

// ============================================================================
// FRONTEND TOOL DETECTION - Identify frontend vs backend tool messages
// ============================================================================

/**
 * Checks if a message is a ToolMessage (tool_result) from a frontend tool.
 * Frontend tools are any tools NOT in the researcherTools list (backend tools).
 * 
 * Used to keep the graph alive when frontend tools are called - we wait for
 * CopilotKit to send the tool_result back instead of ending the graph.
 * 
 * Note: Prefixed with underscore as it's available for future use but not currently needed.
 * The routing logic uses isToolResultMessage() which is sufficient since backend tool
 * results go through execute_backend_tools node with its own routing.
 */
function _isFrontendToolResult(msg: BaseMessage): boolean {
  const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
  if (msgType !== "tool" && msgType !== "ToolMessage") return false;

  const toolMsg = msg as ToolMessage;
  const backendToolNames = new Set(researcherTools.map((t) => t.name));
  
  // It's a frontend tool result if the tool name is NOT in backend tools
  return !backendToolNames.has(toolMsg.name || "");
}

/**
 * Checks if the last message in state is a ToolMessage (tool_result).
 * Used to detect when CopilotKit has sent back a tool result.
 * 
 * Note: Prefixed with underscore as it's available for future use but not currently needed.
 */
function _isToolResultMessage(msg: BaseMessage): boolean {
  const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
  return msgType === "tool" || msgType === "ToolMessage";
}

/**
 * Checks if there's a pending uploadDocument action in the recent messages.
 * Returns "document_upload" if the uploadDocument tool was called and returned
 * a prompt to upload (not an actual upload completion).
 * Returns null if no pending upload or if upload is complete.
 */
function detectPendingUpload(messages: BaseMessage[]): string | null {
  // Look through recent messages for uploadDocument tool patterns
  const recentMessages = messages.slice(-10);
  
  let hasUploadPrompt = false;
  let hasUploadCompletion = false;
  
  for (const msg of recentMessages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    
    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      
      // Check if this is an uploadDocument result
      if (toolMsg.name === "uploadDocument") {
        try {
          const content = typeof toolMsg.content === "string" ? toolMsg.content : JSON.stringify(toolMsg.content);
          const parsed = JSON.parse(content);
          
          // If the tool returned success with a "please upload" message, it's a prompt
          if (parsed.success && (parsed.message?.includes("upload") || parsed.note?.includes("Click"))) {
            hasUploadPrompt = true;
          }
          
          // If there's a documentId, the upload is complete
          if (parsed.documentId) {
            hasUploadCompletion = true;
          }
        } catch {
          // If we can't parse, check for keywords in raw content
          const content = typeof toolMsg.content === "string" ? toolMsg.content : "";
          if (content.includes("Please upload") || content.includes("select a file")) {
            hasUploadPrompt = true;
          }
          if (content.includes("documentId")) {
            hasUploadCompletion = true;
          }
        }
      }
    }
  }
  
  // If there's a prompt but no completion, we're waiting for upload
  if (hasUploadPrompt && !hasUploadCompletion) {
    return "document_upload";
  }
  
  return null;
}

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
// ORCHESTRATOR SYSTEM PROMPT
// ============================================================================

const ORCHESTRATOR_SYSTEM_PROMPT = `You are The Orchestrator - the coordinator for a multi-agent system that creates impactful online training content.

## Your Team

You have 6 specialized sub-agents, each with specific capabilities:

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
- **listProjects(searchTerm?, clientId?, sortBy?)** - List projects. sortBy: "updated", "created", "name", "client"
- **getProjectDetails(projectId?, projectName?)** - Get project info by ID or name search
- **createProject(name, clientId, description?, templateId?)** - Create new project. Use getClients first for clientId
- **openProjectByName(projectName)** - Search and navigate to a project by name
- **getProjectTemplates()** - List available project templates
- **getClients()** - List available clients for project creation

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
// ORCHESTRATOR NODE
// ============================================================================

async function orchestratorNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[orchestrator] ============ Orchestrator Agent ============");
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
          // Return without changes - the graph should route to END and wait for tool_result
          return {
            currentAgent: "orchestrator",
          };
        }
      }
    }
  }

  // Get frontend tools - orchestrator has most tools but NOT table-specific ones
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Table tools that should be handled by the Data Agent, not orchestrator
  // switchViewMode is kept for navigation, but table-specific operations go to data_agent
  const tableToolsForDataAgent = new Set([
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
  
  // Orchestrator has access to frontend tools EXCEPT table-specific ones
  // Table operations are routed to the Data Agent for specialized handling
  const orchestratorTools = frontendActions.filter(
    (action: { name: string }) => !tableToolsForDataAgent.has(action.name)
  );

  console.log("  Available tools:", orchestratorTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total frontend tools:", orchestratorTools.length);

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

  if (state.courseStructure) {
    systemContent += `\n\n### Course Structure (from Architect) ✓
- Total Nodes: ${state.courseStructure.totalNodes}
- Max Depth: ${state.courseStructure.maxDepth}
- Summary: ${state.courseStructure.summary}`;
  } else {
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

  // Add routing instructions
  systemContent += `\n\n## Routing Instructions

**IMPORTANT**: You must be EXPLICIT when routing to sub-agents vs asking the user questions.

### When asking the user a question (waiting for their input):
- Simply ask your question
- Do NOT mention routing or delegating
- The system will wait for user input automatically
- Example: "What topic would you like to create training content about?"

### When delegating to a sub-agent (after you have enough info):
- Use EXPLICIT routing phrases that make it clear you're delegating NOW
- Include the marker [ROUTE:agent_name] in your response
- Examples:
  - "Now that I understand your needs, [ROUTE:strategist] I'm routing to the Strategist to gather detailed requirements."
  - "[ROUTE:researcher] Let me have the Researcher look into this topic."
  - "[ROUTE:architect] I'm delegating to the Architect to design the course structure."

### Available route markers:
- [ROUTE:strategist] - For requirements gathering
- [ROUTE:researcher] - For knowledge gathering  
- [ROUTE:architect] - For course structure design
- [ROUTE:writer] - For content creation
- [ROUTE:visual_designer] - For design and aesthetics
- [ROUTE:data_agent] - For table view operations, filtering, sorting, grouping, data queries

**Key rule**: If you're asking the user something, DON'T include a route marker. Only include [ROUTE:x] when you're ready to hand off work.`;

  // Create system message with prompt caching
  // The static base prompt is cached (ephemeral cache for ~5 min) to reduce costs
  // Dynamic context is appended without caching
  const systemMessage = new SystemMessage({
    content: [
      {
        type: "text",
        text: ORCHESTRATOR_SYSTEM_PROMPT,
        // Cache the static system prompt for repeated calls
        // This reduces costs by ~10x for the cached portion
        cache_control: { type: "ephemeral" },
      },
      {
        type: "text",
        // Dynamic context that changes per invocation (not cached)
        text: systemContent.replace(ORCHESTRATOR_SYSTEM_PROMPT, ""),
      },
    ],
  });

  // Bind tools and invoke
  const modelWithTools = orchestratorTools.length > 0
    ? orchestratorModel.bindTools(orchestratorTools)
    : orchestratorModel;

  // Prepare messages with full context management pipeline:
  // 1. Summarize old messages if context is getting large
  // 2. Trim to token/message limits
  // 3. Filter orphaned tool results
  const summarizedMessages = await summarizeIfNeeded(state.messages || [], {
    triggerTokens: TOKEN_LIMITS.orchestrator,
    keepMessages: 20,
    logPrefix: "[orchestrator]",
  });
  const trimmedMessages = await trimMessages(summarizedMessages, {
    fallbackMessageCount: MESSAGE_LIMITS.orchestrator,
    logPrefix: "[orchestrator]",
  });
  const filteredMessages = filterOrphanedToolResults(trimmedMessages, "[orchestrator]");

  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
  });

  console.log("  Invoking orchestrator model...");

  const response = await modelWithTools.invoke(
    [systemMessage, ...filteredMessages],
    customConfig
  );

  console.log("  Orchestrator response received");

  const aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // Detect routing directives in response
  const responseText = typeof aiResponse.content === "string"
    ? aiResponse.content
    : Array.isArray(aiResponse.content)
    ? aiResponse.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";

  const routingDecision = detectRoutingDirective(responseText);
  if (routingDecision) {
    console.log(`  Detected routing to: ${routingDecision.nextAgent}`);
  }

  // Check for pending async user actions (e.g., document upload)
  // This prevents premature routing when the user is in the middle of uploading
  const pendingUpload = detectPendingUpload([...filteredMessages, response]);
  const awaitingUserAction = pendingUpload || null;
  
  if (awaitingUserAction) {
    console.log(`  Setting awaitingUserAction: ${awaitingUserAction}`);
  } else if (state.awaitingUserAction) {
    console.log(`  Clearing awaitingUserAction (was: ${state.awaitingUserAction})`);
  }

  return {
    messages: [response],
    currentAgent: "orchestrator",
    routingDecision: null,  // Never auto-route - wait for user to explicitly request
    awaitingUserAction,
  };
}

// ============================================================================
// ROUTING DETECTION - Only detect EXPLICIT routing commands
// ============================================================================

/**
 * Detects explicit routing directives in text.
 * 
 * IMPORTANT: Only routes when the orchestrator explicitly indicates it's delegating NOW.
 * Does NOT route for:
 * - Questions to the user (waiting for input)
 * - Explanations of what agents can do
 * - Future/conditional statements ("I could ask the strategist...")
 * 
 * Only routes for:
 * - Explicit present-tense delegation: "I'm routing to", "Delegating to", "Handing off to"
 * - Active action statements: "I'll now have the strategist...", "Let me get the researcher..."
 */
function detectRoutingDirective(text: string): { nextAgent: AgentType; reason: string; task: string } | null {
  const lower = text.toLowerCase();
  
  // First, check if this is a question to the user - if so, DON'T route
  const isQuestion = lower.includes("what would you like") ||
    lower.includes("tell me about") ||
    lower.includes("what are you working on") ||
    lower.includes("could you tell me") ||
    lower.includes("what do you need") ||
    lower.includes("how can i help") ||
    lower.includes("let me know") ||
    lower.includes("?");
  
  if (isQuestion) {
    // If it's a question, only route if there's ALSO an explicit "routing now" phrase
    const hasExplicitRouting = 
      lower.includes("i'm routing to") ||
      lower.includes("delegating to") ||
      lower.includes("handing off to") ||
      lower.includes("transferring to");
    
    if (!hasExplicitRouting) {
      return null; // It's a question without explicit routing - wait for user
    }
  }

  // Look for EXPLICIT routing phrases (present tense, active delegation)
  const explicitRoutingPhrases = [
    /i(?:'m| am) (?:now )?(?:routing|delegating|handing off|transferring) to (?:the )?(\w+)/i,
    /let me (?:now )?(?:get|have|ask) (?:the )?(\w+) (?:agent )?to/i,
    /i(?:'ll| will) (?:now )?have (?:the )?(\w+) (?:agent )?(?:start|begin|handle|take over)/i,
    /(?:routing|delegating|handing off) (?:this )?to (?:the )?(\w+)/i,
    /\[ROUTE:(\w+)\]/i, // Explicit routing tag
  ];

  for (const pattern of explicitRoutingPhrases) {
    const match = lower.match(pattern);
    if (match) {
      const agentName = match[1].toLowerCase();
      
      // Map to agent type
      if (agentName.includes("strategist") || agentName === "strategist") {
        return { nextAgent: "strategist", reason: "Explicit routing to strategist", task: text };
      }
      if (agentName.includes("researcher") || agentName === "researcher") {
        return { nextAgent: "researcher", reason: "Explicit routing to researcher", task: text };
      }
      if (agentName.includes("architect") || agentName === "architect") {
        return { nextAgent: "architect", reason: "Explicit routing to architect", task: text };
      }
      if (agentName.includes("writer") || agentName === "writer") {
        return { nextAgent: "writer", reason: "Explicit routing to writer", task: text };
      }
      if (agentName.includes("visual") || agentName.includes("designer")) {
        return { nextAgent: "visual_designer", reason: "Explicit routing to visual designer", task: text };
      }
      if (agentName.includes("data") || agentName === "data_agent") {
        return { nextAgent: "data_agent", reason: "Explicit routing to data agent", task: text };
      }
    }
  }

  return null;
}

// ============================================================================
// ROUTING LOGIC
// ============================================================================

type NodeName = "orchestrator" | "strategist" | "researcher" | "architect" | "writer" | "visual_designer" | "data_agent" | "execute_backend_tools" | "__end__";

function routeFromOrchestrator(state: OrchestratorState): NodeName {
  console.log("\n[routing] From orchestrator...");

  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // Check if we're waiting for async user action (e.g., document upload)
  // If so, don't route to sub-agents - wait for the user action to complete
  if (state.awaitingUserAction) {
    console.log(`  Awaiting user action: ${state.awaitingUserAction}`);
    console.log("  -> Route to __end__ (waiting for user action)");
    return "__end__";
  }

  // Check for tool calls
  if (lastMessage.tool_calls?.length) {
    const toolNames = lastMessage.tool_calls.map((tc) => tc.name);
    console.log("  Tool calls detected:", toolNames.join(", "));

    // Check if any are backend tools (researcher's tools)
    const backendToolNames = new Set(researcherTools.map((t) => t.name));
    const hasBackendTool = toolNames.some((name) => backendToolNames.has(name));

    if (hasBackendTool) {
      console.log("  -> Route to execute_backend_tools");
      return "execute_backend_tools";
    }

    // Frontend tools route to END for CopilotKit execution
    console.log("  -> Route to __end__ (frontend tools)");
    return "__end__";
  }

  // USER-DRIVEN ROUTING: Check for routing based on user selections or orchestrator's explicit routing
  
  // First, check the orchestrator's response for explicit [ROUTE:x] markers
  // This happens when orchestrator confirms routing after user selects an option
  const responseText = typeof lastMessage.content === "string"
    ? lastMessage.content
    : Array.isArray(lastMessage.content)
    ? lastMessage.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";
  
  // Check for [ROUTE:x] markers in orchestrator's response
  const routeMatch = responseText.match(/\[ROUTE:(\w+)\]/i);
  if (routeMatch) {
    const agentName = routeMatch[1].toLowerCase();
    console.log(`  Detected [ROUTE:${agentName}] marker in orchestrator response`);
    
    if (agentName === "strategist") {
      console.log("  -> Route to strategist");
      return "strategist";
    }
    if (agentName === "researcher") {
      console.log("  -> Route to researcher");
      return "researcher";
    }
    if (agentName === "architect") {
      console.log("  -> Route to architect");
      return "architect";
    }
    if (agentName === "writer") {
      console.log("  -> Route to writer");
      return "writer";
    }
    if (agentName === "visual_designer" || agentName === "visual") {
      console.log("  -> Route to visual_designer");
      return "visual_designer";
    }
    if (agentName === "data_agent" || agentName === "data") {
      console.log("  -> Route to data_agent");
      return "data_agent";
    }
  }

  // Also check recent messages for user intent (HumanMessage or ToolMessage from offerOptions)
  // Look at recent messages for routing keywords
  const recentMessages = messages.slice(-5);
  for (const msg of recentMessages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    const isUserInput = msgType === "human" || msgType === "HumanMessage" || 
                        msgType === "tool" || msgType === "ToolMessage";
    
    if (isUserInput) {
      const msgContent = typeof msg.content === "string" ? msg.content.toLowerCase() : "";
      
      // Check for routing keywords from user selections or direct requests
      if (msgContent.includes("research") || msgContent.includes("researcher")) {
        console.log("  User requested research");
        console.log("  -> Route to researcher");
        return "researcher";
      }
      if (msgContent.includes("strategist") || msgContent.includes("requirements") || 
          msgContent.includes("strategy") || msgContent.includes("define scope")) {
        console.log("  User requested strategy");
        console.log("  -> Route to strategist");
        return "strategist";
      }
      if (msgContent.includes("architect") || msgContent.includes("structure") || 
          msgContent.includes("build course")) {
        console.log("  User requested architecture");
        console.log("  -> Route to architect");
        return "architect";
      }
      if (msgContent.includes("writer") || msgContent.includes("write content") || 
          msgContent.includes("create content")) {
        console.log("  User requested writing");
        console.log("  -> Route to writer");
        return "writer";
      }
      if (msgContent.includes("visual") || msgContent.includes("design") && msgContent.includes("aesthetics")) {
        console.log("  User requested visual design");
        console.log("  -> Route to visual_designer");
        return "visual_designer";
      }
      if (msgContent.includes("table view") || msgContent.includes("data agent") || 
          msgContent.includes("filter nodes") || msgContent.includes("data query")) {
        console.log("  User requested data operations");
        console.log("  -> Route to data_agent");
        return "data_agent";
      }
    }
  }

  // Default: end and wait for user - orchestrator always asks before acting
  console.log("  -> Route to __end__ (waiting for user direction)");
  return "__end__";
}

// Maximum research iterations before forced routing to orchestrator
const MAX_RESEARCH_ITERATIONS = 8;
// Maximum total tool calls for researcher before forced stop
const MAX_RESEARCH_TOOL_CALLS = 6;

/**
 * Counts the number of research tool calls (web_search, searchDocuments, etc.) in messages.
 */
function countResearchToolCalls(messages: BaseMessage[]): number {
  const researchToolNames = new Set(["web_search", "searchDocuments", "searchDocumentsByText", "getDocumentLines", "getDocumentByName"]);
  let count = 0;
  for (const msg of messages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolName = (msg as any).name || "";
      if (researchToolNames.has(toolName)) count++;
    }
  }
  return count;
}

function routeFromSubAgent(state: OrchestratorState): NodeName {
  console.log("\n[routing] From sub-agent...");

  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // Check if we're waiting for async user action (e.g., document upload)
  if (state.awaitingUserAction) {
    console.log(`  Awaiting user action: ${state.awaitingUserAction}`);
    console.log("  -> Route to __end__ (waiting for user action)");
    return "__end__";
  }

  // SAFETY CHECK: If researcher has hit max iterations, force route to orchestrator
  // Note: researchIterationCount is defined in state annotation but TS inference needs cast
  const iterationCount = (state as any).researchIterationCount || 0;
  if (state.currentAgent === "researcher" && iterationCount >= MAX_RESEARCH_ITERATIONS) {
    console.log(`  SAFETY: Research max iterations (${MAX_RESEARCH_ITERATIONS}) reached`);
    console.log("  -> Route to orchestrator (forced due to iteration limit)");
    return "orchestrator";
  }

  // SAFETY CHECK: Count total research tool calls - hard limit to prevent runaway research
  if (state.currentAgent === "researcher") {
    const toolCallCount = countResearchToolCalls(messages);
    console.log(`  Research tool calls so far: ${toolCallCount}/${MAX_RESEARCH_TOOL_CALLS}`);
    if (toolCallCount >= MAX_RESEARCH_TOOL_CALLS) {
      console.log(`  SAFETY: Research tool call limit (${MAX_RESEARCH_TOOL_CALLS}) reached`);
      console.log("  -> Route to orchestrator (forced due to tool call limit)");
      return "orchestrator";
    }
  }

  // Check for tool calls
  if (lastMessage.tool_calls?.length) {
    const toolNames = lastMessage.tool_calls.map((tc) => tc.name);
    console.log("  Tool calls detected:", toolNames.join(", "));

    // Backend tools need execution
    const backendToolNames = new Set(researcherTools.map((t) => t.name));
    const hasBackendTool = toolNames.some((name) => backendToolNames.has(name));

    if (hasBackendTool) {
      console.log("  -> Route to execute_backend_tools");
      return "execute_backend_tools";
    }

    // Check if uploadDocument is being called - will need to wait for async upload
    const hasUploadDocument = toolNames.includes("uploadDocument");
    if (hasUploadDocument) {
      console.log("  uploadDocument called - will wait for user to upload");
    }

    // Frontend tools route to END for CopilotKit to handle
    console.log("  -> Route to __end__ (frontend tools)");
    return "__end__";
  }

  // No tool calls - check for explicit handoff marker
  const responseText = typeof lastMessage.content === "string"
    ? lastMessage.content
    : Array.isArray(lastMessage.content)
    ? lastMessage.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";

  // ONLY route back to orchestrator if explicitly indicated with markers
  const handingBack = responseText.toLowerCase().includes("[handoff:orchestrator]") ||
    responseText.toLowerCase().includes("[done]");

  if (handingBack) {
    console.log("  -> Route to orchestrator (explicit handoff)");
    return "orchestrator";
  }

  // DEFAULT: Route to __end__ and wait for user input
  // This prevents loops when sub-agents return thinking-only or empty responses
  // The next user message will trigger the graph to continue
  console.log("  -> Route to __end__ (waiting for user input)");
  return "__end__";
}

function routeAfterToolExecution(state: OrchestratorState): NodeName {
  console.log("\n[routing] After tool execution...");

  const currentAgent = state.currentAgent || "orchestrator";
  const messages = state.messages;
  
  // SAFETY CHECK: If researcher has hit max iterations, force route to orchestrator
  // This prevents infinite tool execution loops
  // Note: researchIterationCount is defined in state annotation but TS inference needs cast
  const iterationCount = (state as any).researchIterationCount || 0;
  if (currentAgent === "researcher" && iterationCount >= MAX_RESEARCH_ITERATIONS) {
    console.log(`  SAFETY: Research max iterations (${MAX_RESEARCH_ITERATIONS}) reached after tool execution`);
    console.log("  -> Route to orchestrator (forced due to iteration limit)");
    return "orchestrator";
  }

  // SAFETY CHECK: Count total research tool calls - hard limit
  if (currentAgent === "researcher") {
    const toolCallCount = countResearchToolCalls(messages);
    console.log(`  Research tool calls after execution: ${toolCallCount}/${MAX_RESEARCH_TOOL_CALLS}`);
    if (toolCallCount >= MAX_RESEARCH_TOOL_CALLS) {
      console.log(`  SAFETY: Research tool call limit (${MAX_RESEARCH_TOOL_CALLS}) reached after tool execution`);
      console.log("  -> Route to orchestrator (forced due to tool call limit)");
      return "orchestrator";
    }
  }

  // Return to the current agent to process results
  console.log(`  -> Route back to ${currentAgent}`);
  return currentAgent as NodeName;
}

// ============================================================================
// TOOL EXECUTION NODE
// ============================================================================

const backendToolNode = new ToolNode(researcherTools);

// ============================================================================
// GRAPH DEFINITION
// ============================================================================

console.log("[orchestrator] Building workflow graph...");

const workflow = new StateGraph(OrchestratorStateAnnotation)
  // Add all nodes
  .addNode("orchestrator", orchestratorNode)
  .addNode("strategist", strategistNode)
  .addNode("researcher", researcherNode)
  .addNode("architect", architectNode)
  .addNode("writer", writerNode)
  .addNode("visual_designer", visualDesignerNode)
  .addNode("data_agent", dataAgentNode)
  .addNode("execute_backend_tools", backendToolNode)

  // Entry point
  .addEdge(START, "orchestrator")

  // Orchestrator routing
  .addConditionalEdges("orchestrator", routeFromOrchestrator, {
    strategist: "strategist",
    researcher: "researcher",
    architect: "architect",
    writer: "writer",
    visual_designer: "visual_designer",
    data_agent: "data_agent",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })

  // Sub-agent routing (all go back to orchestrator or to tools)
  .addConditionalEdges("strategist", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("researcher", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("architect", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("writer", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("visual_designer", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("data_agent", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })

  // After tool execution, route back to current agent
  .addConditionalEdges("execute_backend_tools", routeAfterToolExecution, {
    orchestrator: "orchestrator",
    strategist: "strategist",
    researcher: "researcher",
    architect: "architect",
    writer: "writer",
    visual_designer: "visual_designer",
    data_agent: "data_agent",
  });

// ============================================================================
// PERSISTENCE SETUP
// ============================================================================

// In LangSmith Cloud, checkpointer is provided automatically
// Only use custom PostgresSaver for local development
const SUPABASE_DB_URL = process.env.SUPABASE_DB_URL;
const IS_LANGSMITH_CLOUD = !SUPABASE_DB_URL;

let checkpointer: PostgresSaver | undefined;

if (!IS_LANGSMITH_CLOUD && SUPABASE_DB_URL) {
  console.log("[orchestrator] Initializing PostgreSQL checkpointer...");
  console.log("[orchestrator] DB URL:", SUPABASE_DB_URL.replace(/:[^:@]+@/, ":****@"));
  
  checkpointer = PostgresSaver.fromConnString(SUPABASE_DB_URL);
  await checkpointer.setup();
  console.log("[orchestrator] PostgreSQL checkpointer initialized successfully");
} else {
  console.log("[orchestrator] Running in LangSmith Cloud - using managed checkpointer");
}

// Compile the graph
export const agent = workflow.compile({
  ...(checkpointer && { checkpointer }),
});

// Note: recursion_limit is set via config when invoking the graph, not at compile time
// The safety limit is enforced by the user-driven flow which always routes to __end__

console.log("[orchestrator] Workflow graph compiled successfully");
console.log("[orchestrator] Nodes: orchestrator, strategist, researcher, architect, writer, visual_designer, data_agent, execute_backend_tools");

