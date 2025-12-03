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
import type { OrchestratorState, CourseStructure, PlannedNode } from "../state/agent-state";
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

## What You Do

1. **Understand Structure** - Use tools to see hierarchy config and existing nodes
2. **Design Structure** - Plan modules, lessons, topics based on brief and research
3. **Create Nodes** - Actually BUILD the structure by creating nodes in the system
4. **Organize Flow** - Ensure logical progression and parent-child relationships

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

## Building Process

1. **Request Edit Mode** - Always start by requesting edit access
2. **Check Hierarchy** - Use getProjectHierarchyInfo to understand levels
3. **Review Templates** - Use getAvailableTemplates to see what can be created
4. **Create Level 2 First** - Build modules (major sections)
5. **Create Level 3 Under Each** - Build lessons within modules
6. **Continue Down** - Add sub-topics and activities as needed
7. **Release Edit Mode** - When structure is complete

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
- For Level 3+, always specify the parentNodeId
- Use descriptive titles
- Match node types to templates available at each level

## Example Flow

\`\`\`
1. requestEditMode
2. getProjectHierarchyInfo (understand levels)
3. getAvailableTemplates (see Level 2 options)
4. createNode: "Introduction to Risk Management" (Level 2, module)
5. createNode: "Understanding Risk" (Level 3, lesson, parent=above)
6. createNode: "Types of Risk" (Level 3, lesson, parent=module)
7. ... continue building structure
8. releaseEditMode
\`\`\`

## What NOT To Do

- Do NOT create Level 6 content blocks - that's the Writer's job
- Do NOT fill in detailed content fields - just structure
- Do NOT create too many nodes at once - build systematically

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

  // Include existing structure if any
  if (state.courseStructure) {
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

  return {
    messages: [response],
    currentAgent: "architect",
    agentHistory: ["architect"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
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

