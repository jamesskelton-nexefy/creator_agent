/**
 * Architect Agent
 *
 * Structures training content for maximum learning impact.
 * Applies frameworks thoughtfully without rigid adherence.
 *
 * Tools (Read-only frontend):
 * - getProjectHierarchyInfo - Understand hierarchy levels and coding
 * - getAvailableTemplates - See what node types can be created
 * - getNodesByLevel - See existing nodes at specific levels
 * - getNodeChildren - Check children of specific nodes
 *
 * Input: Reads projectBrief and researchFindings from state
 * Output: courseStructure in shared state
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState, CourseStructure, PlannedNode } from "../state/agent-state";
import { getCondensedBrief, getCondensedResearch } from "../state/agent-state";

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const architectModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 16000,
  temperature: 0.5, // Balanced for creative structure with practical constraints
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const ARCHITECT_SYSTEM_PROMPT = `You are The Architect - a specialized agent focused on structuring online training for maximum learning impact.

## Your Role

You design course structures that:
1. **Maximize Learning** - Structure content for optimal retention and application
2. **Follow Logical Flow** - Build knowledge progressively, concept upon concept
3. **Balance Depth & Breadth** - Cover enough without overwhelming
4. **Meet Requirements** - Satisfy objectives, regulations, and constraints
5. **Engage Learners** - Create varied, interesting learning experiences

## Your Philosophy

You understand that great training structure goes BEYOND simply mapping frameworks 1:1. Instead, you:
- Use frameworks as guides, not rigid templates
- Prioritize learner experience over bureaucratic compliance
- Create natural learning progressions
- Group related concepts meaningfully
- Include practical application opportunities
- Balance theory with hands-on activities

## Your Tools

You have read-only access to understand the project structure:

### Hierarchy Tools
- **getProjectHierarchyInfo** - Understand available hierarchy levels and coding
- **getAvailableTemplates** - See what node templates can be used
- **getNodesByLevel** - See existing nodes at specific levels
- **getNodeChildren** - Check children of specific nodes

## Course Structure Design

### Hierarchy Levels (typical)
- **Level 2: Modules/Sections** - Major topic areas (3-8 typically)
- **Level 3: Lessons/Topics** - Focused learning units within modules
- **Level 4: Sub-topics** - Detailed breakdowns when needed
- **Level 5: Activities** - Specific learning activities
- **Level 6: Content Blocks** - Actual content pieces

### Design Principles

1. **Chunking** - Break content into digestible pieces (7±2 items per level)
2. **Scaffolding** - Build from simple to complex
3. **Interleaving** - Mix related topics for better retention
4. **Spacing** - Distribute practice across the course
5. **Variation** - Use different content types and activities

## Output Format

Produce a detailed course structure:

\`\`\`json
{
  "summary": "Brief description of the overall structure",
  "rationale": "Why this structure was chosen",
  "maxDepth": 4,
  "totalNodes": 25,
  "nodes": [
    {
      "tempId": "m1",
      "title": "Module Title",
      "nodeType": "module",
      "level": 2,
      "parentTempId": null,
      "description": "What this module covers",
      "objectives": ["Objective 1", "Objective 2"],
      "orderIndex": 1
    },
    {
      "tempId": "m1-l1",
      "title": "Lesson Title",
      "nodeType": "lesson",
      "level": 3,
      "parentTempId": "m1",
      "description": "What this lesson covers",
      "orderIndex": 1
    }
  ]
}
\`\`\`

## Guidelines

- Start by understanding the hierarchy configuration
- Review the project brief for scope and objectives
- Use research findings to inform topic selection
- Create logical parent-child relationships
- Assign meaningful tempIds for reference (e.g., "m1", "m1-l1", "m1-l1-t1")
- Include clear descriptions for each node
- Map learning objectives to specific nodes
- Consider the target audience's prior knowledge
- Plan for variety in content types

Remember: A well-structured course makes the difference between forgettable training and transformative learning.`;

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
  // The architect only uses read-only hierarchy tools
  const frontendActions = state.copilotkit?.actions ?? [];
  const architectTools = frontendActions.filter((action: { name: string }) =>
    ["getProjectHierarchyInfo", "getAvailableTemplates", "getNodesByLevel", "getNodeChildren"].includes(action.name)
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

  // Filter messages for this agent's context
  const recentMessages = (state.messages || []).slice(-12);

  console.log("  Invoking architect model...");

  const response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Architect response received");

  const aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  return {
    messages: [response],
    currentAgent: "architect",
    agentHistory: ["architect"],
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

