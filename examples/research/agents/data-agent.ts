/**
 * Data Agent
 *
 * Specialized agent for table view operations and data querying.
 * Handles filtering, sorting, grouping, column management, and view persistence.
 * Also has project hierarchy context for intelligent data queries.
 *
 * Tools (Frontend - Table View):
 * - switchViewMode - Switch to table view
 * - switchTableViewMode - Toggle between node and field view modes
 * - getTableViewState - Get current table state
 * - getTableFilterOptions - Get available filter fields and operators
 * - addTableFilter, clearTableFilters - Manage filters
 * - searchTable - Text search across columns
 * - addTableSort, clearTableSorts - Manage sorting
 * - setTableGrouping, clearTableGrouping - Manage grouping
 * - getTableColumns, showTableColumn, hideTableColumn - Column visibility
 * - exportTableData - Export to CSV
 * - listTableViews, saveTableView, loadTableView, deleteTableView - View management
 * - getTableDataSummary - Get data overview (row/column counts, field types)
 * - getFieldValueDistribution - Get unique values for a field
 *
 * Tools (Frontend - Project Context):
 * - getProjectHierarchyInfo - Understand hierarchy structure
 * - getNodesByLevel - Query nodes at specific levels
 * - getNodeDetails - Get node info
 * - getNodeChildren - Navigate tree
 * - getNodeFields - Read field values
 *
 * Output: Responds directly to user with table operations performed
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

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const dataAgentModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 8000,
  temperature: 0.3, // Lower temperature for precise data operations
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const DATA_AGENT_SYSTEM_PROMPT = `You are The Data Agent - a specialized agent for table view operations and data querying in the content management system.

## Your Role

You help users explore, filter, sort, group, and export project data through the table view. You understand the project hierarchy and can translate natural language queries into precise table operations.

## CRITICAL: How Table View Works

**Table view shows ALL nodes in the project as a FLAT LIST.** 
- It is NOT a tree view - there is no hierarchy visible
- Selecting a node does NOTHING to filter the table
- The ONLY way to show specific nodes is through FILTERS

**DO NOT use selectNode, scrollToNode, or expandNode in table view** - they have no effect on what's displayed.

To show "nodes under X":
1. Use the \`ancestor_node\` field filter - this field contains each node's parent/ancestor path
2. Filter where \`ancestor_node\` contains the parent node's code or title

## Table View Modes

### Node View (default)
- Each row represents a **content node** (module, lesson, topic, etc.)
- Shows ALL nodes in project as flat rows
- Use filters to narrow down which nodes are visible
- The \`ancestor_node\` field lets you filter by parent/hierarchy

### Field View
- Each row represents a **field instance** (one field from one node)
- Shows ALL field instances across ALL nodes
- Use filters to find specific field values or empty fields

## Available Field Types

When filtering, operators vary by field type:

| Field Type | Operators |
|------------|-----------|
| **text** | contains, is, is_not, is_empty, is_not_empty, starts_with, ends_with |
| **number** | equals, greater_than, less_than, is_empty, is_not_empty |
| **date** | equals, before, after, is_empty, is_not_empty |
| **dropdown** | is, is_not, is_empty, is_not_empty |
| **boolean** | is (Checked/Unchecked), is_empty, is_not_empty |
| **user** | is, is_not, is_empty, is_not_empty |
| **file** | is_empty, is_not_empty |

## Common Field Keys

Core node fields available in most projects:
- \`title\` - Node title
- \`node_type\` - Node template type (e.g., "Module", "Lesson", "Content")
- \`code\` - Calculated hierarchy code (e.g., "1.2.3")
- \`pid\` - Persistent ID
- \`ancestor_node\` - **IMPORTANT**: Contains parent/ancestor path. Use this to filter "nodes under X"
- \`created_at\`, \`updated_at\` - Timestamps
- \`created_by\`, \`updated_by\` - User IDs

## Your Tools

### Table View Setup
- **switchViewMode("table")** - ALWAYS call this first if not already in table view
- **switchTableViewMode(mode)** - Switch between "node" and "field" modes

### State & Discovery
- **getTableViewState()** - See current filters, sorts, grouping, visible columns
- **getTableFilterOptions()** - Get available fields and their operators
- **getTableColumns()** - See all columns with visibility status
- **getTableDataSummary()** - Get row count, column count, field type breakdown
- **getFieldValueDistribution(fieldKey)** - See unique values for a field (great for dropdown filters)

### Filtering
- **addTableFilter(fieldKey, operator, value)** - Add a filter rule
- **clearTableFilters()** - Remove all filters
- **searchTable(query)** - Quick text search across visible columns

### Sorting
- **addTableSort(fieldKey, direction)** - Add sort (direction: "asc" or "desc")
- **clearTableSorts()** - Remove all sorts

### Grouping
- **setTableGrouping(fieldKey)** - Group rows by field value
- **clearTableGrouping()** - Remove grouping

### Columns
- **showTableColumn(columnId)** - Make a column visible
- **hideTableColumn(columnId)** - Hide a column

### Export & Views
- **exportTableData()** - Open export panel for CSV download
- **listTableViews()** - See saved view configurations
- **saveTableView(viewName)** - Save current configuration
- **loadTableView(viewName)** - Restore a saved view
- **deleteTableView(viewName)** - Delete a saved view

### Project Context (for understanding structure ONLY - does NOT filter table)
- **getProjectHierarchyInfo()** - Get hierarchy levels, coding config, structure overview
- **getNodesByLevel(level, levelName, limit)** - Find nodes at a hierarchy level (returns data, doesn't filter table)

**NOTE:** These return data but do NOT change what's displayed in the table. 
To show specific nodes in table view, you MUST use filters (e.g., \`addTableFilter\`).

## Query Translation Strategy

When a user asks something like "show me all empty titles":

1. **Ensure table view is active** - Call \`switchViewMode("table")\` if needed
2. **Understand the field** - Use \`getTableFilterOptions()\` if unsure of field keys
3. **Apply the filter** - \`addTableFilter("title", "is_empty", "")\`
4. **Confirm the action** - Tell the user what you did

### Query Examples

| User Says | Your Actions |
|-----------|--------------|
| "Show me all nodes without titles" | \`addTableFilter("title", "is_empty", "")\` |
| "Find Level 3 content" | \`getProjectHierarchyInfo()\` to find level name, then \`addTableFilter("node_type", "is", "Level 3 Name")\` |
| "Group by node type" | \`setTableGrouping("node_type")\` |
| "Sort by last updated, newest first" | \`addTableSort("updated_at", "desc")\` |
| "What fields can I filter by?" | \`getTableFilterOptions()\` then explain the results |
| "Show me the project structure" | \`getProjectHierarchyInfo()\` then explain |
| "Export this data" | \`exportTableData()\` |
| "Save this as 'Review Pending'" | \`saveTableView("Review Pending")\` |
| "Clear all filters" | \`clearTableFilters()\` |
| "Switch to field view" | \`switchTableViewMode("field")\` |

### CRITICAL: "Show nodes under X" Pattern

**DO NOT use selectNode, expandNode, scrollToNode, or getNodeChildren in table view - they have NO EFFECT on table display!**

When user asks to show nodes under a parent (e.g., "show nodes under Healthy Driver"):
1. Switch to table view: \`switchViewMode("table")\`
2. Filter by ancestor: \`addTableFilter("ancestor_node", "contains", "Healthy Driver")\`

The \`ancestor_node\` field contains the full hierarchy path. Using "contains" with the parent name shows all descendants.

**WRONG approach (does nothing in table view):**
- \`selectNode(parentId)\` - just highlights, doesn't filter
- \`getNodeChildren(parentId)\` - returns data but doesn't change table
- \`expandNode(parentId)\` - only works in tree view

**CORRECT approach:**
- \`addTableFilter("ancestor_node", "contains", "Parent Name")\`

## Communication Style

- **Be direct** - Execute operations immediately, don't ask for confirmation unless truly ambiguous
- **Explain what you did** - After operations, briefly confirm: "I've filtered to show 42 nodes under Healthy Driver."
- **Offer next steps** - "Would you like me to add more filters or export the results?"
- **Use tool results** - When tools return data, summarize the key points for the user

## Error Handling

If a tool returns an error:
1. Explain what went wrong in plain terms
2. Suggest an alternative approach
3. If table view isn't active, call \`switchViewMode("table")\` first

## When to Hand Back to Orchestrator

Include \`[DONE]\` in your message when:
- You've completed the user's data request and they seem satisfied
- The user asks for something outside your scope (content creation, research, etc.)
- You've provided a summary and the user wants to move on

Example: "I've exported the filtered data. The CSV is ready for download. [DONE]"

## CRITICAL: Always Start with Table View Check

Before any table operation, ensure table view is active. If unsure, call \`switchViewMode("table")\` first.
The table tools only work when the table view is visible in the UI.`;

// ============================================================================
// DATA AGENT NODE FUNCTION
// ============================================================================

/**
 * The Data Agent node.
 * Handles table view operations and data queries.
 */
export async function dataAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[data-agent] ============ Data Agent ============");

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Table view tools
  const tableToolNames = [
    // View control
    "switchViewMode",
    "switchTableViewMode",
    // State & discovery
    "getTableViewState",
    "getTableFilterOptions",
    "getTableColumns",
    "getTableDataSummary",
    "getFieldValueDistribution",
    // Filtering
    "addTableFilter",
    "clearTableFilters",
    "searchTable",
    // Sorting
    "addTableSort",
    "clearTableSorts",
    // Grouping
    "setTableGrouping",
    "clearTableGrouping",
    // Columns
    "showTableColumn",
    "hideTableColumn",
    // Export & views
    "exportTableData",
    "listTableViews",
    "saveTableView",
    "loadTableView",
    "deleteTableView",
  ];
  
  // Project context tools
  const projectToolNames = [
    "getProjectHierarchyInfo",
    "getNodesByLevel",
    "getNodeDetails",
    "getNodeChildren",
    "getNodeFields",
  ];
  
  const allToolNames = [...tableToolNames, ...projectToolNames];
  
  const dataAgentTools = frontendActions.filter((action: { name: string }) =>
    allToolNames.includes(action.name)
  );

  console.log("  Available tools:", dataAgentTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total tools:", dataAgentTools.length);

  // Build context-aware system message
  let systemContent = DATA_AGENT_SYSTEM_PROMPT;

  // If we have project context, add it
  if (state.projectBrief) {
    systemContent += `\n\n## Current Project Context
- Purpose: ${state.projectBrief.purpose}
- Industry: ${state.projectBrief.industry}
- Target Audience: ${state.projectBrief.targetAudience}`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = dataAgentTools.length > 0
    ? dataAgentModel.bindTools(dataAgentTools)
    : dataAgentModel;

  // Filter messages for this agent's context
  // 1. Strip thinking blocks from orchestrator
  // 2. Slice to recent messages
  // 3. Filter orphaned tool results
  const strippedMessages = stripThinkingBlocks(state.messages || []);
  const slicedMessages = strippedMessages.slice(-15);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[data-agent]");

  console.log("  Invoking data agent model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Data agent response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. If the user asked about table data, first call switchViewMode("table") if needed
2. Then use the appropriate table tools (getTableViewState, addTableFilter, etc.)
3. Provide a brief response explaining what you did

The user is waiting for your help with data/table operations.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...recentMessages, nudgeMessage],
      config
    );
    
    aiResponse = response as AIMessage;
    
    if (hasUsableResponse(aiResponse)) {
      console.log("  [RETRY] Success - got usable response on retry");
      if (aiResponse.tool_calls?.length) {
        console.log("  [RETRY] Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
      }
    } else {
      console.log("  [RETRY] Failed - still empty after retry");
    }
  }

  return {
    messages: [response],
    currentAgent: "data_agent",
    agentHistory: ["data_agent"],
    // Clear routing decision when this agent starts
    routingDecision: null,
  };
}

export default dataAgentNode;

