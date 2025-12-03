/**
 * Strategist Agent
 *
 * Discovers purpose, objectives, scope, and constraints for training projects.
 * Often the first agent called to establish the project brief.
 *
 * Tools:
 * - askClarifyingQuestions (frontend HITL) - Sequential questions with options
 * - offerOptions (frontend HITL) - Present choices to user
 * - getProjectHierarchyInfo - Understand project structure when scoping
 * - listProjects - Check existing projects for reference
 *
 * Output: projectBrief in shared state
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState, ProjectBrief } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  stripThinkingBlocks,
  hasUsableResponse,
} from "../utils";

// Message filtering and thinking block stripping now handled by centralized utils

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const strategistModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7, // Thinking disabled - strategist is conversational, not analytical
});

// Empty response detection now handled by centralized utils/context-management.ts

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const STRATEGIST_SYSTEM_PROMPT = `You are The Strategist - a specialized agent focused on discovering the purpose, objectives, scope, and constraints for online training projects.

## Your Role

You are the first step in creating impactful training content. Your job is to deeply understand:
1. **Purpose** - Why is this training being created? What problem does it solve?
2. **Objectives** - What specific outcomes should learners achieve?
3. **Scope** - What topics and depth are in/out of scope?
4. **Constraints** - Time limits, budget, technical requirements, compliance needs
5. **Target Audience** - Who are the learners? What's their background?
6. **Industry Context** - What industry or domain is this for?

## Your Tools

You have access to two Human-in-the-Loop tools:

### askClarifyingQuestions
Use this to ask the user a series of questions (up to 5) to gather information.
- Each question should have 2-5 clear, distinct options
- Questions are presented one at a time
- Order questions from general to specific
- Make options mutually exclusive when possible

### offerOptions
Use this for single-choice decisions when you need the user to pick between approaches.

## Strategy Session Flow

1. **Greet and Orient** - Briefly explain your role
2. **IMMEDIATELY Ask Key Questions** - You MUST call askClarifyingQuestions right away to gather:
   - Primary training purpose/goal
   - Target audience (role, experience level)
   - Industry/domain context
   - Scope preferences (broad overview vs deep dive)
   - Any constraints or requirements
3. **Synthesize** - Process user responses into a structured project brief
4. **Confirm** - Summarize back and get confirmation

## CRITICAL: Tool Usage - READ THIS CAREFULLY

**YOU MUST CALL askClarifyingQuestions ON YOUR FIRST TURN.**

DO NOT:
- Just think about asking questions
- Just write text explaining what you'll do
- Produce only thinking blocks with no tool call
- Say "I will ask..." without actually calling the tool

DO:
- IMMEDIATELY call the askClarifyingQuestions tool
- Include a brief greeting in text AND call the tool in the same response

Example of CORRECT behavior:
1. Output a brief greeting text: "Hi! I'm the Strategist..."
2. ALSO call askClarifyingQuestions tool with your questions

If you produce thinking but no tool call or text, the user sees NOTHING and the conversation hangs.

## Output Format

After gathering information, structure your findings as a project brief that includes:
- purpose: Clear statement of the training's primary goal
- objectives: 3-5 specific, measurable learning outcomes
- inScope: Topics and areas to cover
- outOfScope: What we're explicitly not covering
- constraints: Time, technical, regulatory limitations
- targetAudience: Detailed learner persona
- industry: Industry or domain context
- regulations: Any compliance requirements (if applicable)

## Guidelines

- Be conversational but efficient - don't ask unnecessary questions
- Listen for implicit requirements in user responses
- Consider regulatory/compliance needs based on industry
- Think about practical application of the training
- Default to sensible assumptions when not specified
- Always validate your understanding before finalizing

## Communication Flow

**On your FIRST turn:**
- ALWAYS call askClarifyingQuestions tool - this is mandatory
- You can include a brief text greeting along with the tool call

**When asking follow-up questions:**
- Use askClarifyingQuestions or offerOptions tools - NOT plain text
- Tools provide better UI with clickable options

**When you've completed gathering requirements:**
- Summarize the project brief in text
- Include [DONE] at the end of your message to hand back to the orchestrator
- Example: "Here's the project brief I've compiled... [DONE]"

Remember: Your output directly shapes everything that follows. A clear, well-defined project brief leads to better research, structure, and content.`;

// ============================================================================
// STRATEGIST NODE FUNCTION
// ============================================================================

/**
 * The Strategist agent node.
 * Gathers project requirements through clarifying questions.
 */
export async function strategistNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[strategist] ============ Strategist Agent ============");
  console.log("  Current project brief:", state.projectBrief ? "exists" : "none");

  // Get frontend tools from CopilotKit state
  // The strategist uses conversation tools + minimal read tools for context
  const frontendActions = state.copilotkit?.actions ?? [];
  const strategistTools = frontendActions.filter((action: { name: string }) =>
    [
      // Conversation tools
      "askClarifyingQuestions",
      "offerOptions",
      // Context tools (understand what exists)
      "getProjectHierarchyInfo",
      "listProjects",
    ].includes(action.name)
  );

  console.log("  Available tools:", strategistTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = STRATEGIST_SYSTEM_PROMPT;

  // If we already have a partial brief, include it
  if (state.projectBrief) {
    systemContent += `\n\n## Current Project Brief (partial)
    
The following information has already been gathered:
- Purpose: ${state.projectBrief.purpose || "Not yet defined"}
- Industry: ${state.projectBrief.industry || "Not yet defined"}
- Target Audience: ${state.projectBrief.targetAudience || "Not yet defined"}
- Objectives: ${state.projectBrief.objectives?.length || 0} defined

Continue gathering any missing information or refine what's been captured.`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = strategistTools.length > 0
    ? strategistModel.bindTools(strategistTools)
    : strategistModel;

  // Filter messages for this agent's context (keep it focused)
  // 1. Strip thinking blocks (strategist has thinking disabled but receives from orchestrator with thinking enabled)
  // 2. Slice to recent messages
  // 3. Filter orphaned tool results AFTER slicing (slicing can create new orphans!)
  const strippedMessages = stripThinkingBlocks(state.messages || []);
  const slicedMessages = strippedMessages.slice(-10);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[strategist]");

  console.log("  Invoking strategist model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Strategist response received");
  
  // Log response details for debugging
  let aiResponse = response as AIMessage;
  
  // Check tool calls
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  } else {
    console.log("  No tool calls in response");
  }
  
  // Check content type
  const content = aiResponse.content;
  if (typeof content === "string") {
    console.log("  Response type: string, length:", content.length);
  } else if (Array.isArray(content)) {
    const blockTypes = content.map((b: any) => b?.type || typeof b).join(", ");
    console.log("  Response type: array, blocks:", blockTypes);
    
    // Check for text blocks
    const textBlocks = content.filter((b: any) => b?.type === "text");
    if (textBlocks.length === 0) {
      console.log("  WARNING: No text blocks in response - only thinking");
    } else {
      const textLength = textBlocks.reduce((sum: number, b: any) => sum + (b.text?.length || 0), 0);
      console.log("  Text content length:", textLength);
    }
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    // Add a nudge message to prompt the model to take action
    // Note: Using HumanMessage because SystemMessage must be first in the array
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. Greet the user briefly
2. IMMEDIATELY call the askClarifyingQuestions tool to start gathering project requirements

The user is waiting for your response.`,
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
    currentAgent: "strategist",
    agentHistory: ["strategist"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
  };
}

/**
 * Parses a strategist's text response to extract a project brief.
 * Called by the orchestrator when the strategist completes its work.
 */
export function parseProjectBrief(content: string): ProjectBrief | null {
  try {
    // Look for JSON block in the response
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      return validateProjectBrief(parsed);
    }

    // Look for structured sections
    const brief: Partial<ProjectBrief> = {};

    // Extract purpose
    const purposeMatch = content.match(/(?:purpose|goal):\s*(.+?)(?:\n|$)/i);
    if (purposeMatch) brief.purpose = purposeMatch[1].trim();

    // Extract objectives (list items)
    const objectivesMatch = content.match(/objectives?:?\s*((?:\n\s*[-*]\s*.+)+)/i);
    if (objectivesMatch) {
      brief.objectives = objectivesMatch[1]
        .split("\n")
        .filter((line) => line.trim().match(/^[-*]/))
        .map((line) => line.replace(/^[-*]\s*/, "").trim());
    }

    // Extract target audience
    const audienceMatch = content.match(/(?:target audience|learners?|audience):\s*(.+?)(?:\n|$)/i);
    if (audienceMatch) brief.targetAudience = audienceMatch[1].trim();

    // Extract industry
    const industryMatch = content.match(/(?:industry|domain|sector):\s*(.+?)(?:\n|$)/i);
    if (industryMatch) brief.industry = industryMatch[1].trim();

    // Only return if we have minimum required fields
    if (brief.purpose && brief.targetAudience) {
      return validateProjectBrief(brief);
    }

    return null;
  } catch (error) {
    console.error("[strategist] Failed to parse project brief:", error);
    return null;
  }
}

/**
 * Validates and fills in defaults for a project brief.
 */
function validateProjectBrief(input: Partial<ProjectBrief>): ProjectBrief {
  return {
    purpose: input.purpose || "Training purpose not specified",
    objectives: input.objectives || [],
    inScope: input.inScope || [],
    outOfScope: input.outOfScope || [],
    constraints: input.constraints || [],
    targetAudience: input.targetAudience || "General learners",
    industry: input.industry || "General",
    regulations: input.regulations,
    notes: input.notes,
  };
}

export default strategistNode;

