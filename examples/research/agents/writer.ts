/**
 * Writer Agent
 *
 * Creates the ACTUAL CONTENT for training courses (Level 6 content blocks).
 * Works within the structure built by the Architect.
 * Writes engaging, educational training material based on research and design guidelines.
 *
 * Tools (Full CRUD + Navigation + Media):
 * - requestEditMode, releaseEditMode - Edit lock management
 * - createNode, getNodeTemplateFields, updateNodeFields, getNodeFields - Content CRUD
 * - getProjectHierarchyInfo, getAvailableTemplates, getNodesByLevel, getNodeDetails - Navigation
 * - searchMicroverse, attachMicroverseToNode - Media integration
 *
 * Input: Reads courseStructure, researchFindings, visualDesign from state
 * Output: Level 6 content nodes with actual training content
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState, ContentOutput, PlannedNode } from "../state/agent-state";
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

const writerModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7,
});

// Empty response detection now handled by centralized utils/context-management.ts

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const WRITER_SYSTEM_PROMPT = `You are The Writer - you CREATE the actual training content that learners interact with.

## Your Role

You work within the structure built by the Architect and create Level 6 content blocks - the actual training material. Your content:

1. **Engages** - Captures and maintains learner attention
2. **Educates** - Clearly explains concepts and procedures  
3. **Applies** - Connects theory to practical application
4. **Adapts** - Matches the learner's level and industry context
5. **Achieves** - Directly supports learning objectives

## What You Do vs The Architect

- **Architect** builds the skeleton: modules, lessons, topics (Levels 2-5)
- **You** fill in the content: content blocks, activities, assessments (Level 6)

The structure should already exist when you start. Your job is to create the actual training content within that structure.

## Your Tools

### Edit Mode (REQUIRED)
- **requestEditMode** - Request edit lock before making changes
- **releaseEditMode** - Release edit lock when done

### Content Creation
- **createNode** - Create Level 6 content blocks
- **getNodeTemplateFields** - Get field schema to understand what to write
- **updateNodeFields** - Update content in existing nodes
- **getNodeFields** - Read current content from nodes

### Navigation (Find where to add content)
- **getProjectHierarchyInfo** - Understand hierarchy levels
- **getAvailableTemplates** - See what content templates exist
- **getNodesByLevel** - Find parent nodes to add content under
- **getNodeDetails** - Get detailed info about a node
- **getNodeChildren** - See what content already exists

### Media Integration
- **searchMicroverse** - Find relevant images, videos, assets
- **attachMicroverseToNode** - Attach media to your content

## Content Writing Process

1. **Request Edit Mode** - Always start by requesting edit access
2. **Navigate Structure** - Use getNodesByLevel to find where to add content
3. **Check Templates** - Use getAvailableTemplates to see Level 6 content types
4. **Get Field Schema** - Use getNodeTemplateFields to know what fields to fill
5. **Write Content** - Create engaging, educational content
6. **Create Content Node** - Use createNode with initialFields populated
7. **Add Media** - Search and attach relevant media if appropriate
8. **Repeat** - Continue for each content block needed
9. **Release Edit Mode** - When done writing

## Writing Guidelines

### Tone & Voice
- Match the visual design's writing tone guidelines
- Be consistent throughout the course
- Use active voice when possible
- Address learners directly (second person: "you")

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

## Content Types You Create

- **Text Blocks** - Explanatory content, descriptions
- **Lists** - Steps, bullet points, key takeaways
- **Examples** - Case studies, scenarios, demonstrations
- **Activities** - Interactive elements, reflection prompts
- **Summaries** - Key points, recaps

## Guidelines

- Create content nodes ONE AT A TIME - wait for success
- Find the parent node FIRST before creating under it
- Fill in ALL relevant fields when creating
- Match content to the learning objectives
- Keep content scannable with headers and bullets
- Use concrete examples over abstract theory
- Consider adding media to enhance engagement

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
  // The writer uses CRUD for content + navigation to find structure + media tools
  const frontendActions = state.copilotkit?.actions ?? [];
  const writerTools = frontendActions.filter((action: { name: string }) =>
    [
      // Edit mode management
      "requestEditMode",
      "releaseEditMode",
      // Content CRUD
      "createNode",
      "getNodeTemplateFields",
      "updateNodeFields",
      "getNodeFields",
      // Navigation (find where to add content)
      "getProjectHierarchyInfo",
      "getAvailableTemplates",
      "getNodesByLevel",
      "getNodeDetails",
      "getNodeChildren",
      // Media integration
      "searchMicroverse",
      "attachMicroverseToNode",
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

  // Filter messages for this agent's context - filter orphans first, then slice
  // Filter AFTER slicing - slicing can create new orphans by removing AI messages with tool_use
  const slicedMessages = (state.messages || []).slice(-10);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[writer]");

  console.log("  Invoking writer model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Writer response received");

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
1. Call requestEditMode to get edit access
2. Call getNodesByLevel to find where to add content
3. Start creating content nodes

The user is waiting for you to write content.`,
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
    currentAgent: "writer",
    agentHistory: ["writer"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
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

