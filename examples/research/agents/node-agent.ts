/**
 * Node Agent
 *
 * Specialized agent for node operations, including creating nodes,
 * updating fields, managing edit mode, and exploring node templates.
 *
 * Tools (Frontend):
 * - Edit Mode: requestEditMode, releaseEditMode, checkEditStatus
 * - Node Read: getProjectHierarchyInfo, getNodeChildren, getNodeDetails,
 *              getNodesByLevel, getNodeFields
 * - Node Write: createNode, updateNodeFields
 * - Templates: getAvailableTemplates, getNodeTemplateFields,
 *              listAllNodeTemplates, listFieldTemplates, getTemplateDetails
 * - Navigation: scrollToNode, expandAllNodes, collapseAllNodes
 *
 * Output: Responds directly to user with node operations
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

const nodeAgentModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 8000,
  temperature: 0.4,
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const NODE_AGENT_SYSTEM_PROMPT = `You are The Node Agent - a specialized agent for node and template operations.

## Your Role

You help users create and manage content nodes within projects. You handle:
- Creating new nodes with appropriate templates
- Updating node fields and content
- Managing edit mode (required for changes)
- Exploring hierarchy structure and templates
- Navigating the node tree

## CRITICAL: Edit Mode

**You MUST request edit mode before any write operation!**

1. Before createNode or updateNodeFields → Call requestEditMode() first
2. After completing changes → Call releaseEditMode()
3. If edit mode fails → Explain that someone else may be editing

## Your Tools

### Edit Mode Control
- **requestEditMode()** - Request edit lock (REQUIRED before writing)
- **releaseEditMode()** - Release edit lock when done
- **checkEditStatus()** - Check current edit status

### Node Information (Read)
- **getProjectHierarchyInfo()** - Get hierarchy levels, coding config
- **getNodeChildren(nodeId?)** - Get children of a node
- **getNodeDetails(nodeId?)** - Get detailed node info
- **getNodesByLevel(level?, levelName?, limit?)** - Find nodes at a hierarchy level
- **getNodeFields(nodeId?)** - Read current field values

### Node Operations (Write - REQUIRES EDIT MODE)
- **createNode(templateId, title, parentNodeId?, initialFields?)** - Create a new node
  - First call getAvailableTemplates(parentNodeId) to get valid templates
  - initialFields uses assignment IDs from getNodeTemplateFields
- **updateNodeFields(nodeId?, fieldUpdates)** - Update node fields
  - fieldUpdates format: { assignmentId: value }
  - Get assignment IDs from getNodeTemplateFields or getNodeFields

### Template Discovery
- **getAvailableTemplates(parentNodeId?)** - Get templates valid for creating under a parent
- **getNodeTemplateFields(templateId)** - Get field schema with assignment IDs
- **listAllNodeTemplates(searchTerm?, nodeType?)** - Browse all node templates
- **listFieldTemplates(fieldType?)** - Browse field templates
- **getTemplateDetails(templateId, templateType?)** - Get full template details

### Tree Navigation
- **scrollToNode(nodeId)** - Scroll to make a node visible
- **expandAllNodes()** - Expand entire tree
- **collapseAllNodes()** - Collapse entire tree

## Workflow Examples

### Creating a New Node
1. Call getProjectHierarchyInfo() to understand structure
2. Call getAvailableTemplates(parentNodeId) to see valid templates
3. Call getNodeTemplateFields(templateId) to understand fields
4. Call requestEditMode() - CRITICAL!
5. Call createNode(templateId, title, parentNodeId, initialFields)
6. Call releaseEditMode()

### Updating Node Content
1. Call getNodeFields(nodeId) to see current values
2. Call requestEditMode() - CRITICAL!
3. Call updateNodeFields(nodeId, { assignmentId: newValue })
4. Call releaseEditMode()

### Exploring Templates
1. Call listAllNodeTemplates() to see available templates
2. Call getTemplateDetails(templateId) for full details
3. Call listFieldTemplates() to see field types

## Communication Style

- Always check edit mode status before writing
- Explain what you're creating/updating
- Summarize the result after operations
- If edit fails, explain why and suggest alternatives

## When to Hand Back

Include \`[DONE]\` in your response when:
- You've completed the requested node operations
- You've provided the template/hierarchy information
- The user wants to do something outside node management

Example: "I've created the new module with 3 lessons. [DONE]"`;

// ============================================================================
// NODE AGENT NODE FUNCTION
// ============================================================================

/**
 * The Node Agent node.
 * Handles node creation, updates, and template operations.
 */
export async function nodeAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[node-agent] ============ Node Agent ============");

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Node operation tools
  const nodeToolNames = TOOL_CATEGORIES.node;
  
  const nodeAgentTools = frontendActions.filter((action: { name: string }) =>
    nodeToolNames.includes(action.name)
  );

  console.log("  Available tools:", nodeAgentTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total tools:", nodeAgentTools.length);

  // Build context-aware system message
  let systemContent = NODE_AGENT_SYSTEM_PROMPT;

  // If we have project context, add it
  if (state.projectBrief) {
    systemContent += `\n\n## Current Project Context
- Purpose: ${state.projectBrief.purpose}
- Industry: ${state.projectBrief.industry}
- Target Audience: ${state.projectBrief.targetAudience}`;
  }

  // Add course structure if available
  if (state.courseStructure) {
    systemContent += `\n\n## Course Structure Plan
The architect has designed the following structure. Use this as guidance when creating nodes:
${JSON.stringify(state.courseStructure, null, 2).substring(0, 2000)}...`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = nodeAgentTools.length > 0
    ? nodeAgentModel.bindTools(nodeAgentTools)
    : nodeAgentModel;

  // Filter messages for this agent's context
  const strippedMessages = stripThinkingBlocks(state.messages || []);
  const slicedMessages = strippedMessages.slice(-15);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[node-agent]");

  console.log("  Invoking node agent model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Node agent response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. If creating/updating nodes, first call requestEditMode()
2. Then call the appropriate tool (createNode, updateNodeFields, etc.)
3. Provide a helpful response explaining what you did

The user is waiting for your help with node operations.`,
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
    currentAgent: "node_agent",
    agentHistory: ["node_agent"],
    routingDecision: null,
  };
}

export default nodeAgentNode;








