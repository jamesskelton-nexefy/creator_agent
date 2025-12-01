/**
 * Writer Agent
 *
 * Creates Level 6 content nodes with industry-appropriate training content.
 * Considers industry, learner persona, and purpose to write perfect training content.
 *
 * Tools (Full CRUD):
 * - requestEditMode, releaseEditMode - Edit lock management
 * - createNode, getNodeTemplateFields, updateNodeFields, getNodeFields - Node CRUD
 *
 * Input: Reads courseStructure, researchFindings, visualDesign from state
 * Output: writtenContent tracking created nodes
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState, ContentOutput, PlannedNode } from "../state/agent-state";
import { getCondensedBrief, getCondensedResearch } from "../state/agent-state";

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const writerModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 16000,
  temperature: 0.6, // Creative but consistent
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const WRITER_SYSTEM_PROMPT = `You are The Writer - a specialized agent focused on creating impactful training content.

## Your Role

You write Level 6 content nodes - the actual training material that learners interact with. Your content:

1. **Engages** - Captures and maintains learner attention
2. **Educates** - Clearly explains concepts and procedures
3. **Applies** - Connects theory to practical application
4. **Adapts** - Matches the learner's level and industry context
5. **Achieves** - Directly supports learning objectives

## Your Tools

You have full CRUD access for content creation:

### Edit Mode
- **requestEditMode** - Request edit lock before making changes
- **releaseEditMode** - Release edit lock when done

### Node Operations
- **createNode** - Create new content nodes
- **getNodeTemplateFields** - Get field schema for a template
- **updateNodeFields** - Update content in existing nodes
- **getNodeFields** - Read current content from nodes

## Content Writing Process

1. **Request Edit Mode** - Always start by requesting edit access
2. **Review Context** - Understand the planned structure, research, and design
3. **Get Template Fields** - Understand what fields need content
4. **Write Content** - Create engaging, educational content
5. **Create/Update Nodes** - Save content to the system
6. **Release Edit Mode** - Free the lock for others

## Writing Guidelines

### Tone & Voice
- Match the visual design's writing tone guidelines
- Be consistent throughout the course
- Use active voice when possible
- Address learners directly (typically second person)

### Content Structure
- Start with a clear learning objective or hook
- Break complex topics into digestible chunks
- Use examples relevant to the industry
- Include practical applications
- End with key takeaways or next steps

### Engagement Techniques
- Use questions to prompt thinking
- Include scenarios and case studies
- Vary content types (text, lists, examples)
- Connect to real-world situations
- Build on prior knowledge

### Industry Adaptation
- Use industry-specific terminology appropriately
- Reference relevant regulations when applicable
- Include industry examples and scenarios
- Consider the learner's work context

## Output Tracking

Track created content:

\`\`\`json
{
  "nodeId": "created-node-uuid",
  "tempId": "m1-l1-c1",
  "title": "Content Block Title",
  "fieldsWritten": ["content", "summary"],
  "createdAt": "2024-01-15T10:30:00Z",
  "notes": "Any relevant notes"
}
\`\`\`

## Guidelines

- Create nodes ONE AT A TIME - wait for success before proceeding
- Always check if edit mode is active before creating
- Match content to the learning objectives
- Keep content scannable with headers and bullets
- Include concrete examples over abstract theory
- Write for your specific audience (not generic learners)
- Maintain consistent formatting throughout

Remember: Great content transforms learners. Every word should serve the learning experience.`;

// ============================================================================
// WRITER NODE FUNCTION
// ============================================================================

/**
 * The Writer agent node.
 * Creates content nodes based on the course structure.
 */
export async function writerNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[writer] ============ Writer Agent ============");
  console.log("  Course structure available:", state.courseStructure ? "yes" : "no");
  console.log("  Nodes written so far:", state.writtenContent?.length || 0);
  console.log("  Nodes created in session:", state.createdNodes?.length || 0);

  // Get frontend tools from CopilotKit state
  // The writer uses edit mode and node CRUD tools
  const frontendActions = state.copilotkit?.actions ?? [];
  const writerTools = frontendActions.filter((action: { name: string }) =>
    [
      "requestEditMode",
      "releaseEditMode",
      "createNode",
      "getNodeTemplateFields",
      "updateNodeFields",
      "getNodeFields",
    ].includes(action.name)
  );

  console.log("  Available tools:", writerTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = WRITER_SYSTEM_PROMPT;

  // Include project brief for context
  if (state.projectBrief) {
    const condensedBrief = getCondensedBrief(state.projectBrief);
    systemContent += `\n\n## Project Context\n\n${condensedBrief}`;
  }

  // Include research for content accuracy
  if (state.researchFindings) {
    const condensedResearch = getCondensedResearch(state.researchFindings);
    systemContent += `\n\n## Research to Inform Content\n\n${condensedResearch}`;
  }

  // Include visual design for tone
  if (state.visualDesign) {
    systemContent += `\n\n## Writing Style Guidelines

**Tone**: ${state.visualDesign.writingTone.tone}
**Voice**: ${state.visualDesign.writingTone.voice}
**Complexity**: ${state.visualDesign.writingTone.complexity}
**Typography Style**: ${state.visualDesign.typography.style}

${state.visualDesign.notes ? `Additional notes: ${state.visualDesign.notes}` : ""}`;
  }

  // Include course structure with progress
  if (state.courseStructure) {
    const plannedNodes = state.courseStructure.nodes;
    const writtenTempIds = new Set(state.writtenContent?.map((w) => w.tempId) || []);
    const remaining = plannedNodes.filter((n) => !writtenTempIds.has(n.tempId));

    systemContent += `\n\n## Course Structure to Write

**Total Planned**: ${plannedNodes.length} nodes
**Already Written**: ${writtenTempIds.size} nodes
**Remaining**: ${remaining.length} nodes

### Nodes to Write:
${remaining
  .slice(0, 10)
  .map((n) => `- ${n.tempId}: "${n.title}" (${n.nodeType}, level ${n.level})`)
  .join("\n")}${remaining.length > 10 ? `\n... and ${remaining.length - 10} more` : ""}`;
  }

  // Track already created nodes
  if (state.createdNodes && state.createdNodes.length > 0) {
    systemContent += `\n\n## Already Created Nodes (DO NOT RECREATE)
    
${state.createdNodes.map((n) => `- "${n.title}" (ID: ${n.nodeId.substring(0, 8)}...)`).join("\n")}`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = writerTools.length > 0
    ? writerModel.bindTools(writerTools)
    : writerModel;

  // Filter messages for this agent's context
  const recentMessages = (state.messages || []).slice(-10);

  console.log("  Invoking writer model...");

  const response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Writer response received");

  const aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  return {
    messages: [response],
    currentAgent: "writer",
    agentHistory: ["writer"],
  };
}

/**
 * Extracts content output tracking from tool result messages.
 */
export function extractContentOutput(
  toolName: string,
  toolResult: any,
  plannedNode?: PlannedNode
): ContentOutput | null {
  if (toolName !== "createNode") return null;

  try {
    const result = typeof toolResult === "string" ? JSON.parse(toolResult) : toolResult;

    if (result.success && result.newNodeId) {
      return {
        nodeId: result.newNodeId,
        tempId: plannedNode?.tempId || "unknown",
        title: result.title || plannedNode?.title || "Unknown",
        fieldsWritten: result.fieldsSet > 0 ? ["content"] : [],
        createdAt: new Date().toISOString(),
        notes: result.templateName ? `Template: ${result.templateName}` : undefined,
      };
    }
  } catch (error) {
    console.error("[writer] Failed to extract content output:", error);
  }

  return null;
}

/**
 * Gets the next planned node that hasn't been written yet.
 */
export function getNextNodeToWrite(
  structure: PlannedNode[],
  writtenContent: ContentOutput[]
): PlannedNode | null {
  const writtenTempIds = new Set(writtenContent.map((w) => w.tempId));

  // Find first unwritten node, prioritizing by level then order
  const remaining = structure
    .filter((n) => !writtenTempIds.has(n.tempId))
    .sort((a, b) => {
      if (a.level !== b.level) return a.level - b.level;
      return a.orderIndex - b.orderIndex;
    });

  return remaining[0] || null;
}

/**
 * Generates a writing brief for a specific planned node.
 */
export function generateWritingBrief(
  node: PlannedNode,
  structure: PlannedNode[],
  research?: string
): string {
  // Find parent and siblings for context
  const parent = node.parentTempId
    ? structure.find((n) => n.tempId === node.parentTempId)
    : null;

  const siblings = structure.filter(
    (n) => n.parentTempId === node.parentTempId && n.tempId !== node.tempId
  );

  let brief = `## Writing Brief: ${node.title}

**Type**: ${node.nodeType}
**Level**: ${node.level}
**Description**: ${node.description}`;

  if (node.objectives?.length) {
    brief += `\n\n**Learning Objectives**:\n${node.objectives.map((o) => `- ${o}`).join("\n")}`;
  }

  if (parent) {
    brief += `\n\n**Context**: Part of "${parent.title}"`;
  }

  if (siblings.length > 0) {
    brief += `\n\n**Related Topics**:\n${siblings.map((s) => `- ${s.title}`).join("\n")}`;
  }

  if (research) {
    brief += `\n\n**Relevant Research**:\n${research}`;
  }

  return brief;
}

export default writerNode;

