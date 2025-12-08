/**
 * Architect Agent
 *
 * Builds the STRUCTURE of training courses (Levels 2-5).
 * Creates modules, lessons, topics, and sub-topics.
 * Does NOT write final content - that's the Writer's job.
 *
 * Tools (Full CRUD for structure):
 * - requestEditMode, releaseEditMode - Edit lock management
 * - createNode - Create structural nodes (modules, lessons, topics)
 * - getProjectHierarchyInfo - Understand hierarchy levels and coding
 * - getAvailableTemplates - See what node types can be created
 * - getNodeTemplateFields - Get field schema for templates
 * - getNodesByLevel - See existing nodes at specific levels
 * - getNodeChildren - Check children of specific nodes
 * - getNodeDetails - Get detailed node information
 * - listAllNodeTemplates - See all available templates
 *
 * Input: Reads projectBrief and researchFindings from state
 * Output: Creates structural nodes in the project
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState, CourseStructure, PlannedNode, PlannedStructure, CreatedNode } from "../state/agent-state";
import { getCondensedBrief, getCondensedResearch } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  hasUsableResponse,
} from "../utils";

// Message filtering now handled by centralized utils/context-management.ts

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const architectModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7,
});

// Empty response detection now handled by centralized utils/context-management.ts

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const ARCHITECT_SYSTEM_PROMPT = `You are The Architect - you BUILD the structure of online training courses.

## Your Role

You CREATE the structural framework of courses by building nodes at Levels 2-5:
- **Level 2**: Modules/Sections - Major topic areas
- **Level 3**: Lessons/Topics - Focused learning units
- **Level 4**: Sub-topics - Detailed breakdowns
- **Level 5**: Activities - Specific learning activities

**IMPORTANT**: You do NOT write the final content (Level 6 content blocks). That's the Writer's job. You build the skeleton; they fill in the content.

## TWO-PHASE PROCESS

You work in two distinct phases to ensure plans survive interruptions:

### PHASE 1: PLANNING (Output plan BEFORE creating anything)

1. Understand the hierarchy and templates available
2. Design the complete structure based on brief and research
3. **OUTPUT YOUR PLAN AS JSON** with the marker \`[PLAN READY]\`

The plan is saved to state and can be resumed if interrupted!

### PHASE 2: EXECUTION (Create nodes from the plan)

1. Request edit mode
2. Create nodes ONE AT A TIME from your plan
3. Track progress - note which nodes are created
4. Release edit mode when complete
5. Mark \`[STRUCTURE COMPLETE]\` when done

## PLAN OUTPUT FORMAT

Before creating ANY nodes, output your complete plan:

\`\`\`json
{
  "summary": "Brief description of the course structure",
  "rationale": "Why this structure works for the learning objectives",
  "nodes": [
    {
      "tempId": "mod-1",
      "title": "Module 1: Introduction",
      "nodeType": "module",
      "level": 2,
      "parentTempId": null,
      "description": "Overview and foundations",
      "orderIndex": 0
    },
    {
      "tempId": "les-1-1",
      "title": "What is Risk?",
      "nodeType": "lesson",
      "level": 3,
      "parentTempId": "mod-1",
      "description": "Basic risk concepts",
      "orderIndex": 0
    }
  ]
}
\`\`\`
[PLAN READY]

## RESUMING FROM INTERRUPTION

If you see existing plannedStructure or createdNodes in the context:
1. **DO NOT recreate** nodes that already exist
2. Check which tempIds have already been executed
3. Continue from where you left off
4. Skip nodes that match existing createdNodes by title/parent

## Your Tools

### Edit Mode (REQUIRED before creating)
- **requestEditMode** - Request edit lock before making changes
- **releaseEditMode** - Release edit lock when done

### Node Creation
- **createNode** - Create structural nodes (modules, lessons, topics, activities)
- **getNodeTemplateFields** - Get field schema for templates

### Understanding Structure
- **getProjectHierarchyInfo** - Understand hierarchy levels and coding
- **getAvailableTemplates** - See what node templates can be used
- **listAllNodeTemplates** - See ALL templates available
- **getNodesByLevel** - See existing nodes at specific levels
- **getNodeChildren** - Check children of specific nodes
- **getNodeDetails** - Get detailed node information

## Design Principles

1. **Chunking** - 5-7 items per parent (not too many children)
2. **Scaffolding** - Build from simple to complex
3. **Logical Flow** - Each level should progress naturally
4. **Clear Naming** - Descriptive titles that indicate content
5. **Balanced Depth** - Don't go deeper than necessary

## Node Creation Guidelines

When creating nodes:
- Create ONE node at a time - wait for success before proceeding
- Start with Level 2 (modules) - they have no parent
- For Level 3+, always specify the parentNodeId from a CREATED node
- Use descriptive titles
- Match node types to templates available at each level

## What NOT To Do

- Do NOT create Level 6 content blocks - that's the Writer's job
- Do NOT fill in detailed content fields - just structure
- Do NOT create too many nodes at once - build systematically
- Do NOT skip the planning phase - always output [PLAN READY] first

Remember: A well-structured course makes the difference between forgettable training and transformative learning. Build the skeleton that the Writer will bring to life.`;

// ============================================================================
// ARCHITECT NODE FUNCTION
// ============================================================================

/**
 * The Architect agent node.
 * Designs course structure based on project brief and research.
 */
export async function architectNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[architect] ============ Architect Agent ============");
  console.log("  Project brief available:", state.projectBrief ? "yes" : "no");
  console.log("  Research findings available:", state.researchFindings ? "yes" : "no");

  // Get frontend tools from CopilotKit state
  // The architect uses CRUD tools to build structure + read tools to understand hierarchy
  const frontendActions = state.copilotkit?.actions ?? [];
  const architectTools = frontendActions.filter((action: { name: string }) =>
    [
      // Edit mode management
      "requestEditMode",
      "releaseEditMode",
      // Node creation
      "createNode",
      "getNodeTemplateFields",
      // Structure understanding
      "getProjectHierarchyInfo",
      "getAvailableTemplates",
      "listAllNodeTemplates",
      "getNodesByLevel",
      "getNodeChildren",
      "getNodeDetails",
    ].includes(action.name)
  );

  console.log("  Available tools:", architectTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = ARCHITECT_SYSTEM_PROMPT;

  // Include project brief
  if (state.projectBrief) {
    const condensedBrief = getCondensedBrief(state.projectBrief);
    systemContent += `\n\n## Project Brief\n\n${condensedBrief}`;
  } else {
    systemContent += `\n\n## Note\nNo project brief is available. Consider requesting strategy discovery first.`;
  }

  // Include research findings
  if (state.researchFindings) {
    const condensedResearch = getCondensedResearch(state.researchFindings);
    systemContent += `\n\n## Research Findings\n\n${condensedResearch}`;
  }

  // Include existing PLANNED structure for resumption (critical for crash recovery)
  if (state.plannedStructure) {
    const executed = Object.keys(state.plannedStructure.executedNodes || {}).length;
    const total = state.plannedStructure.nodes.length;
    const remaining = state.plannedStructure.nodes.filter(
      n => !state.plannedStructure!.executedNodes[n.tempId]
    );
    
    systemContent += `\n\n## RESUMING FROM EXISTING PLAN
    
**Status**: ${state.plannedStructure.executionStatus} (${executed}/${total} nodes created)

**Your plan is already saved. DO NOT output a new [PLAN READY].**

**Remaining nodes to create:**
${remaining.map(n => `- ${n.tempId}: "${n.title}" (${n.nodeType}, level ${n.level})`).join('\n')}

**Already created (tempId → nodeId):**
${Object.entries(state.plannedStructure.executedNodes).map(([tempId, nodeId]) => `- ${tempId} → ${nodeId}`).join('\n') || '(none yet)'}

Continue creating the remaining nodes. Use the actual nodeId (not tempId) when specifying parentNodeId for children.`;
  }

  // Include existing created nodes for duplicate prevention
  if (state.createdNodes && state.createdNodes.length > 0) {
    systemContent += `\n\n## Already Created Nodes (DO NOT recreate)
    
${state.createdNodes.map(n => `- "${n.title}" (nodeId: ${n.nodeId}, template: ${n.templateName})`).join('\n')}

Check this list before creating any node to avoid duplicates.`;
  }

  // Include existing FINAL structure if any (for refinement)
  if (state.courseStructure && !state.plannedStructure) {
    systemContent += `\n\n## Existing Structure (to refine)\n
Current structure has ${state.courseStructure.totalNodes} nodes across ${state.courseStructure.maxDepth} levels.
Summary: ${state.courseStructure.summary}

You may refine or extend this existing structure.`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = architectTools.length > 0
    ? architectModel.bindTools(architectTools)
    : architectModel;

  // Filter messages for this agent's context - filter orphans first, then slice
  // Filter AFTER slicing - slicing can create new orphans by removing AI messages with tool_use
  const slicedMessages = (state.messages || []).slice(-12);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[architect]");

  console.log("  Invoking architect model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Architect response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    // Note: Using HumanMessage because SystemMessage must be first in the array
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. Call getProjectHierarchyInfo to understand the structure
2. Call getAvailableTemplates to see what you can create
3. Start designing and creating nodes

The user is waiting for you to build the course structure.`,
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

  // Extract response text for parsing
  const responseText = typeof aiResponse.content === "string"
    ? aiResponse.content
    : Array.isArray(aiResponse.content)
    ? aiResponse.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";

  // Check for PLAN READY marker (Phase 1 output)
  const isPlanReady = responseText.toLowerCase().includes("[plan ready]");
  
  // Check for structure completion markers (Phase 2 complete)
  const isStructureComplete = responseText.toLowerCase().includes("[structure complete]") ||
    responseText.toLowerCase().includes("[done]");
  
  // Parse planned structure on [PLAN READY]
  let parsedPlannedStructure: PlannedStructure | null = null;
  if (isPlanReady && !state.plannedStructure) {
    console.log("  [architect] Plan ready detected - parsing planned structure");
    parsedPlannedStructure = parsePlannedStructure(responseText);
    if (parsedPlannedStructure) {
      console.log("  [architect] Parsed planned structure:", {
        totalNodes: parsedPlannedStructure.totalNodes,
        maxDepth: parsedPlannedStructure.maxDepth,
        summary: parsedPlannedStructure.summary?.substring(0, 50) + "...",
      });
    } else {
      console.log("  [architect] WARNING: Could not parse planned structure from response");
    }
  }

  // Parse course structure on completion
  let parsedStructure: CourseStructure | null = null;
  if (isStructureComplete) {
    console.log("  [architect] Structure completion detected - parsing final course structure");
    parsedStructure = parseCourseStructure(responseText);
    if (parsedStructure) {
      console.log("  [architect] Parsed course structure:", {
        totalNodes: parsedStructure.totalNodes,
        maxDepth: parsedStructure.maxDepth,
        summary: parsedStructure.summary?.substring(0, 50) + "...",
      });
    }
  }

  // Extract created nodes from tool call results
  const newCreatedNodes = extractCreatedNodesFromToolCalls(aiResponse, state.messages || []);
  if (newCreatedNodes.length > 0) {
    console.log("  [architect] Extracted created nodes:", newCreatedNodes.map(n => n.title).join(", "));
  }

  // Update plannedStructure execution tracking if we have new created nodes
  let updatedPlannedStructure: PlannedStructure | null = null;
  if (state.plannedStructure && newCreatedNodes.length > 0) {
    const newExecutedNodes: Record<string, string> = {};
    for (const created of newCreatedNodes) {
      // Match by title to find the tempId
      const matchingPlanned = state.plannedStructure.nodes.find(
        n => n.title.toLowerCase() === created.title.toLowerCase()
      );
      if (matchingPlanned) {
        newExecutedNodes[matchingPlanned.tempId] = created.nodeId;
      }
    }
    
    if (Object.keys(newExecutedNodes).length > 0) {
      const allExecuted = { ...state.plannedStructure.executedNodes, ...newExecutedNodes };
      const totalPlanned = state.plannedStructure.nodes.length;
      const totalExecuted = Object.keys(allExecuted).length;
      
      updatedPlannedStructure = {
        ...state.plannedStructure,
        executedNodes: allExecuted,
        executionStatus: totalExecuted >= totalPlanned ? "completed" : "in_progress",
      };
      console.log(`  [architect] Updated plan execution: ${totalExecuted}/${totalPlanned} nodes`);
    }
  }

  return {
    messages: [response],
    currentAgent: "architect",
    agentHistory: ["architect"],
    routingDecision: null,
    // Include parsed planned structure if available (Phase 1)
    ...(parsedPlannedStructure && { plannedStructure: parsedPlannedStructure }),
    // Include updated planned structure execution tracking
    ...(updatedPlannedStructure && { plannedStructure: updatedPlannedStructure }),
    // Include parsed course structure if available (Phase 2 complete)
    ...(parsedStructure && { courseStructure: parsedStructure }),
    // Include new created nodes
    ...(newCreatedNodes.length > 0 && { createdNodes: newCreatedNodes }),
  };
}

/**
 * Parses an architect's text response to extract course structure.
 */
export function parseCourseStructure(content: string): CourseStructure | null {
  try {
    // Look for JSON block
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      return validateCourseStructure(parsed);
    }
    return null;
  } catch (error) {
    console.error("[architect] Failed to parse course structure:", error);
    return null;
  }
}

function validateCourseStructure(input: Partial<CourseStructure>): CourseStructure {
  const nodes = (input.nodes || []).map((n, idx) => validatePlannedNode(n, idx));
  
  return {
    summary: input.summary || "Course structure",
    rationale: input.rationale || "",
    nodes,
    totalNodes: nodes.length,
    maxDepth: Math.max(...nodes.map((n) => n.level), 2),
  };
}

function validatePlannedNode(input: Partial<PlannedNode>, index: number): PlannedNode {
  return {
    tempId: input.tempId || `node-${index}`,
    title: input.title || `Node ${index + 1}`,
    nodeType: input.nodeType || "content",
    level: input.level || 2,
    parentTempId: input.parentTempId || null,
    templateId: input.templateId,
    description: input.description || "",
    objectives: input.objectives,
    orderIndex: input.orderIndex ?? index,
  };
}

/**
 * Parses an architect's [PLAN READY] response to extract planned structure.
 * This is the pre-creation plan that can be resumed if interrupted.
 */
export function parsePlannedStructure(content: string): PlannedStructure | null {
  try {
    // Look for JSON block before [PLAN READY]
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      const nodes = (parsed.nodes || []).map((n: Partial<PlannedNode>, idx: number) => validatePlannedNode(n, idx));
      
      return {
        summary: parsed.summary || "Course structure plan",
        rationale: parsed.rationale || "",
        nodes,
        totalNodes: nodes.length,
        maxDepth: Math.max(...nodes.map((n: PlannedNode) => n.level), 2),
        plannedAt: new Date().toISOString(),
        executionStatus: "planned",
        executedNodes: {},
      };
    }
    return null;
  } catch (error) {
    console.error("[architect] Failed to parse planned structure:", error);
    return null;
  }
}

/**
 * Extracts created node information from tool call results in the message history.
 * Looks for successful createNode tool calls and their results.
 */
function extractCreatedNodesFromToolCalls(aiResponse: AIMessage, messages: BaseMessage[]): CreatedNode[] {
  const createdNodes: CreatedNode[] = [];
  
  // Look for createNode tool calls in the current response
  const createNodeCalls = aiResponse.tool_calls?.filter(tc => tc.name === "createNode") || [];
  
  if (createNodeCalls.length === 0) {
    return createdNodes;
  }

  // Find corresponding tool results in recent messages
  for (const toolCall of createNodeCalls) {
    const toolCallId = toolCall.id;
    if (!toolCallId) continue;

    // Look for the tool result in messages
    for (const msg of messages) {
      const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
      if (msgType === 'tool' || msgType === 'ToolMessage') {
        const toolMsg = msg as any;
        if (toolMsg.tool_call_id === toolCallId) {
          try {
            // Parse the tool result to extract node info
            const content = typeof toolMsg.content === 'string' ? toolMsg.content : JSON.stringify(toolMsg.content);
            
            // Look for nodeId in the result
            const nodeIdMatch = content.match(/nodeId['":\s]+['"]?([a-zA-Z0-9-]+)/);
            if (nodeIdMatch) {
              const args = toolCall.args as { title?: string; parentNodeId?: string; templateName?: string };
              createdNodes.push({
                nodeId: nodeIdMatch[1],
                title: args.title || 'Unknown',
                parentNodeId: args.parentNodeId || null,
                templateName: args.templateName || 'unknown',
              });
            }
          } catch (e) {
            // Ignore parsing errors
          }
        }
      }
    }
  }

  return createdNodes;
}

/**
 * Converts a course structure to a visual tree representation.
 */
export function structureToTree(structure: CourseStructure): string {
  const lines: string[] = [];
  const nodeMap = new Map(structure.nodes.map((n) => [n.tempId, n]));

  // Find root nodes (no parent)
  const roots = structure.nodes.filter((n) => !n.parentTempId);

  function renderNode(node: PlannedNode, prefix: string, isLast: boolean) {
    const connector = isLast ? "└── " : "├── ";
    lines.push(`${prefix}${connector}${node.title} (${node.nodeType})`);

    // Find children
    const children = structure.nodes
      .filter((n) => n.parentTempId === node.tempId)
      .sort((a, b) => a.orderIndex - b.orderIndex);

    const newPrefix = prefix + (isLast ? "    " : "│   ");
    children.forEach((child, idx) => {
      renderNode(child, newPrefix, idx === children.length - 1);
    });
  }

  roots.forEach((root, idx) => {
    renderNode(root, "", idx === roots.length - 1);
  });

  return lines.join("\n");
}

export default architectNode;

