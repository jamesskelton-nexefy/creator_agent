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
 *
 * Key insight: wrapModelCall middleware can intercept model requests
 * and inject tools dynamically at runtime.
 */

import "dotenv/config";
import * as z from "zod";
import { MemorySaver } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { createAgent, createMiddleware } from "langchain";

// ============================================================================
// COPILOTKIT TOOLS INJECTION MIDDLEWARE
// ============================================================================

/**
 * Middleware that captures CopilotKit frontend tools from state and
 * injects them into model calls so the agent can use them directly.
 */
const copilotKitToolsMiddleware = createMiddleware({
  name: "CopilotKitToolsInjection",
  stateSchema: z.object({
    // CopilotKit populates this when invoking the agent
    copilotkit: z.object({
      actions: z.array(z.any()).optional(),
      context: z.array(z.any()).optional(),
    }).optional(),
    // Internal: store captured tools
    _copilotKitTools: z.array(z.any()).optional(),
  }),
  
  // Capture CopilotKit tools before each model call
  beforeModel: (state) => {
    const tools = (state as any).copilotkit?.actions ?? [];
    console.log(`[CopilotKitToolsMiddleware] Captured ${tools.length} frontend tools`);
    
    if (tools.length > 0) {
      const toolNames = tools.slice(0, 5).map((t: any) => t.name || t.function?.name).join(", ");
      console.log(`[CopilotKitToolsMiddleware] Sample tools: ${toolNames}...`);
    }
    
    return {
      _copilotKitTools: tools,
    };
  },

  // Inject CopilotKit tools into the model call
  wrapModelCall: async (request, handler) => {
    // Get captured tools from state
    const capturedTools = (request as any).state?._copilotKitTools ?? 
                          (request as any).runtime?.state?._copilotKitTools ?? [];
    
    console.log(`[CopilotKitToolsMiddleware] Injecting ${capturedTools.length} tools into model call`);
    
    if (capturedTools.length > 0) {
      // Get existing tools from the request
      const existingTools = (request as any).tools ?? [];
      
      // Combine existing tools with CopilotKit tools
      const allTools = [...existingTools, ...capturedTools];
      
      console.log(`[CopilotKitToolsMiddleware] Total tools: ${allTools.length} (${existingTools.length} existing + ${capturedTools.length} CopilotKit)`);
      
      // Create modified request with all tools
      // The exact API depends on langchain version, try multiple approaches
      try {
        if (typeof request.replace === 'function') {
          // Use replace if available
          return handler(request.replace({ tools: allTools }));
        } else {
          // Modify request directly
          (request as any).tools = allTools;
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

## Your Capabilities

You have direct access to frontend tools that allow you to interact with the UI. These tools are provided by CopilotKit and execute on the frontend.

### Navigation & View Control
- **switchViewMode(mode)** - Change view: "document", "list", "graph", or "table"
- **navigateToProject(projectId)** - Navigate to a specific project
- **goToProjectsList()** - Return to the projects list
- **selectNode(nodeId)** - Select a node in the tree
- **scrollToNode(nodeId)** - Scroll to make a node visible
- **expandNode(nodeId)** / **collapseNode(nodeId)** - Expand/collapse nodes
- **toggleDetailPane()** - Show/hide the detail panel
- **showNotification(message, type)** - Show a toast notification

### Edit Mode (REQUIRED before creating/updating nodes)
- **requestEditMode()** - Request edit lock before making changes
- **releaseEditMode()** - Release the edit lock when done
- **checkEditStatus()** - Check current edit status

### Project Management
- **listProjects(searchTerm?, clientId?, sortBy?)** - List projects
- **getProjectDetails(projectId?, projectName?)** - Get project info
- **createProject(name, clientId, description?, templateId?)** - Create new project
- **openProjectByName(projectName)** - Search and navigate to project
- **getProjectTemplates()** - List available project templates
- **getClients()** - List available clients

### Node Operations
- **getProjectHierarchyInfo()** - Get hierarchy structure info
- **getNodeChildren(nodeId?)** - Get children of a node
- **getNodeDetails(nodeId?)** - Get detailed node info
- **getNodesByLevel(level?, levelName?, limit?)** - Find nodes at a level
- **getAvailableTemplates(parentNodeId?)** - Get templates for creating nodes
- **getNodeTemplateFields(templateId)** - Get field schema for a template
- **getNodeFields(nodeId?)** - Read current field values
- **createNode(templateId, title, parentNodeId?, initialFields?)** - Create a node
- **updateNodeFields(nodeId?, fieldUpdates)** - Update node fields

### Table View Operations
- **switchTableViewMode(mode)** - Switch table view mode
- **getTableViewState()** - Get current table state
- **addTableFilter/clearTableFilters** - Manage filters
- **addTableSort/clearTableSorts** - Manage sorting
- **setTableGrouping/clearTableGrouping** - Manage grouping
- **exportTableData()** - Export to CSV

### User Interaction
- **askClarifyingQuestions(questions)** - Ask questions with options
- **offerOptions(title, options)** - Present choices to user
- **requestPlanApproval(plan)** - Get approval for a plan
- **requestActionApproval(action)** - Confirm a sensitive action

### Document & Media
- **uploadDocument(category, instructions?)** - Trigger document upload
- **searchMicroverse(query?)** - Search media assets
- **attachMicroverseToNode/detachMicroverseFromNode** - Manage media attachments

### Framework Mapping
- **listFrameworks(category?, status?)** - List frameworks
- **getFrameworkDetails(frameworkId)** - Get framework info
- **searchASQAUnits(query)** - Search ASQA units
- **mapCriteriaToNode(nodeId, criteriaId)** - Map criteria to node

## CRITICAL: You MUST call tools to take action

When the user asks you to do something, you MUST call the appropriate tool. DO NOT just describe what you would do.

### Examples:
- "switch to list view" → CALL switchViewMode({ mode: "list" })
- "show me projects" → CALL listProjects({})
- "go to graph view" → CALL switchViewMode({ mode: "graph" })
- "create a new node" → First CALL requestEditMode(), then CALL createNode(...)

**ALWAYS call the tool. NEVER just describe the action.**

## Communication Style

1. Be conversational and helpful
2. Execute tool calls immediately when asked
3. Summarize results after tool execution
4. Offer next steps and suggestions`;

// ============================================================================
// CREATE AGENT
// ============================================================================

console.log("[orchestrator-deep] Creating agent...");

// Persistence setup
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

// Create the agent with middleware that injects CopilotKit tools
export const agent = createAgent({
  model: "claude-sonnet-4-20250514",
  systemPrompt: ORCHESTRATOR_SYSTEM_PROMPT,
  tools: [], // CopilotKit tools will be injected via middleware
  middleware: [copilotKitToolsMiddleware],
  checkpointer,
});

console.log("[orchestrator-deep] Agent created successfully");
console.log("[orchestrator-deep] Pattern: createAgent with CopilotKit tools injection middleware");
console.log("[orchestrator-deep] Tools: Dynamically injected from CopilotKit state");
