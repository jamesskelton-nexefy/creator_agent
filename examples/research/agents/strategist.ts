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
 * - listFrameworks - Browse available frameworks in the system
 * - getFrameworkDetails - Get details of a specific framework
 * - searchASQAUnits - Search TGA (training.gov.au) for ASQA units
 * - listDocuments - Check uploaded documents (may include framework docs)
 *
 * Output: projectBrief in shared state (including frameworks)
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";
import type { OrchestratorState, ProjectBrief, AgentWorkState, StrategistPhase, ActiveTask } from "../state/agent-state";
import { STRATEGIST_PHASES, generateTaskContext } from "../state/agent-state";

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
7. **Frameworks & Standards** - What training packages, competency frameworks, or regulatory standards apply?

## Your Tools

### Conversation Tools (Human-in-the-Loop)

**askClarifyingQuestions** - Ask the user a series of questions (up to 5) to gather information.
- Each question should have 2-5 clear, distinct options
- Questions are presented one at a time
- Order questions from general to specific
- Make options mutually exclusive when possible

**offerOptions** - For single-choice decisions when you need the user to pick between approaches.

### Framework & Standards Tools

**listFrameworks** - Browse frameworks already linked to the project or available in the system.

**getFrameworkDetails** - Get detailed information about a specific framework, including its units/competencies.

**searchASQAUnits** - Search training.gov.au (TGA) for ASQA training package units. Use this when:
- User mentions a qualification (e.g., "Certificate III in Heavy Vehicle")
- User mentions a unit code (e.g., "TLID0015")
- User works in a regulated industry (transport, health, construction, etc.)
- Training needs to align with national competency standards

**listDocuments** - Check uploaded documents which may include framework documents, compliance guides, or standards.

### Spreadsheet Framework Upload Tools

**uploadFrameworkCSV** - Prompt the user to upload a CSV or Excel file (.csv, .xlsx, .xls) containing custom framework/competency data. Use when:
- User has their own competency framework in spreadsheet format
- User mentions uploading a CSV, Excel, or spreadsheet
- User has internal standards not in ASQA

**analyzeFrameworkCSV** - After file upload, analyze the structure and suggest column mappings.

**createFrameworkFromCSV** - Create a framework from the uploaded file with specified column mappings.

### Context Tools

**getProjectHierarchyInfo** - Understand existing project structure when scoping.
**listProjects** - Check existing projects for reference.

## Strategy Session Flow

1. **Greet and Orient** - Briefly explain your role
2. **IMMEDIATELY Ask Key Questions** - You MUST call askClarifyingQuestions right away to gather:
   - Primary training purpose/goal
   - Target audience (role, experience level)
   - Industry/domain context
   - Scope preferences (broad overview vs deep dive)
   - Any constraints or requirements
   - **Compliance/Framework needs** - Does this need to align with a training package or standards?
3. **Capture Frameworks** - Based on user responses:
   - If user mentions qualifications/units, use searchASQAUnits to find relevant ASQA units
   - If user mentions compliance/standards, use listFrameworks to check existing frameworks
   - If user uploaded documents, use listDocuments to check for framework documents
   - Ask about specific competencies they want to cover
4. **Synthesize** - Process user responses into a structured project brief
5. **Confirm** - Summarize back and get confirmation (including any frameworks identified)

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
- frameworks: Any relevant training packages, standards, or competency frameworks including:
  - Name of the framework/training package
  - Type (training_package, asqa_unit, custom, uploaded)
  - Specific units or competencies to cover
  - Notes on why it's relevant

### Framework Examples
- If building heavy vehicle training: Search for "TLI Transport and Logistics" training package
- If building health training: Look for HLT Health or CHC Community Services units
- If user mentions a unit code like "TLID0015": Search for that specific unit
- If user uploaded a "Load Restraint Guide": Note it as an uploaded framework document

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
1. Output the project brief as a JSON code block with this EXACT format:
   \`\`\`json
   {
     "purpose": "The primary goal of the training",
     "objectives": ["Learning objective 1", "Learning objective 2"],
     "inScope": ["Topic 1", "Topic 2"],
     "outOfScope": ["Excluded topic 1"],
     "constraints": ["Time limit", "Budget", "Technical requirements"],
     "targetAudience": "Description of learners",
     "industry": "Industry name"
   }
   \`\`\`
2. Include [BRIEF COMPLETE] after the JSON block
3. DO NOT call saveMemory - state handles this automatically. Your JSON output is parsed and stored in agent state, which is automatically passed to other agents.

**Example completion:**
"Based on our discussion, here's the project brief:
\`\`\`json
{
  "purpose": "Train new employees on safety procedures",
  "objectives": ["Identify hazards", "Apply PPE correctly"],
  ...
}
\`\`\`
[BRIEF COMPLETE]"

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

  // Determine current phase from awaitingUserAction state
  const workState = state.awaitingUserAction;
  const currentPhase: StrategistPhase = (workState?.agent === "strategist" && workState?.phase) 
    ? (workState.phase as StrategistPhase)
    : "gathering_requirements";  // Default to first phase
  
  const phaseConfig = STRATEGIST_PHASES[currentPhase];
  console.log(`  Current phase: ${currentPhase} - ${phaseConfig.description}`);
  console.log(`  Allowed tools: ${phaseConfig.allowedTools.join(", ") || "none"}`);

  // Get frontend tools from CopilotKit state
  // PHASE-GATING: Only allow tools permitted in the current phase
  const frontendActions = state.copilotkit?.actions ?? [];
  const allStrategistTools = [
    // Conversation tools
    "askClarifyingQuestions",
    "offerOptions",
    // Framework & Standards tools
    "listFrameworks",
    "getFrameworkDetails", 
    "getFrameworkItems",
    "searchASQAUnits",
    "listDocuments",
    "importASQAUnit",
    // CSV Framework Upload tools
    "uploadFrameworkCSV",
    "analyzeFrameworkCSV",
    "createFrameworkFromCSV",
    // Context tools (understand what exists)
    "getProjectHierarchyInfo",
    "listProjects",
  ];
  
  // Filter to only tools allowed in current phase
  const strategistTools = frontendActions.filter((action: { name: string }) =>
    allStrategistTools.includes(action.name) && phaseConfig.allowedTools.includes(action.name)
  );

  console.log("  Available tools (phase-filtered):", strategistTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build context-aware system message
  let systemContent = STRATEGIST_SYSTEM_PROMPT;
  
  // Add task context for continuity across context trimming
  const taskContext = generateTaskContext(state);
  if (taskContext) {
    systemContent += `\n\n${taskContext}`;
  }
  
  // Add phase-specific instructions
  systemContent += `\n\n## CURRENT PHASE: ${currentPhase.toUpperCase()}

**You are in the "${currentPhase}" phase.** ${phaseConfig.description}

ALLOWED TOOLS IN THIS PHASE: ${phaseConfig.allowedTools.join(", ") || "NONE - output your final brief"}

${currentPhase === "gathering_requirements" ? `
### Phase Instructions
1. Use askClarifyingQuestions to gather ALL requirements FIRST
2. DO NOT call searchASQAUnits or other search tools yet
3. Wait for the user to answer ALL questions before proceeding
4. When you have enough information, output: [PHASE: searching_references]
` : ""}
${currentPhase === "searching_references" ? `
### Phase Instructions
1. Now you can search for relevant ASQA units and frameworks
2. Use searchASQAUnits to find relevant competency units
3. Use listFrameworks/getFrameworkDetails if needed
4. When searches are complete, output: [PHASE: creating_brief]
` : ""}
${currentPhase === "creating_brief" ? `
### Phase Instructions
1. NO TOOL CALLS - just output the final project brief
2. Include all gathered requirements and any relevant ASQA units found
3. Output the brief as a JSON code block
4. End with [BRIEF COMPLETE]
` : ""}`;

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

  // Configure CopilotKit for proper tool emission (emits tool calls to frontend)
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
    emitMessages: true,
  });

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    customConfig
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
      customConfig
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

  // Check for phase transition markers
  const phaseTransitionMatch = responseText.match(/\[PHASE:\s*(\w+)\]/i);
  let nextPhase: StrategistPhase | null = null;
  if (phaseTransitionMatch) {
    const requestedPhase = phaseTransitionMatch[1].toLowerCase() as StrategistPhase;
    if (requestedPhase in STRATEGIST_PHASES) {
      nextPhase = requestedPhase;
      console.log(`  [strategist] Phase transition detected: ${currentPhase} -> ${nextPhase}`);
    }
  }

  // Check for brief completion markers
  const isBriefComplete = responseText.toLowerCase().includes("[brief complete]") ||
    responseText.toLowerCase().includes("[done]");
  
  // Parse project brief on completion
  let parsedBrief: ProjectBrief | null = null;
  if (isBriefComplete) {
    console.log("  [strategist] Brief completion detected - parsing project brief");
    parsedBrief = parseProjectBrief(responseText);
    if (parsedBrief) {
      console.log("  [strategist] Parsed project brief:", {
        purpose: parsedBrief.purpose?.substring(0, 50) + "...",
        objectivesCount: parsedBrief.objectives?.length || 0,
        industry: parsedBrief.industry,
      });
    } else {
      console.log("  [strategist] WARNING: Could not parse structured project brief from response");
    }
  }

  // Check if response has HITL tool calls that require user interaction
  const HITL_TOOLS = ["askClarifyingQuestions", "offerOptions"];
  const hasHITLToolCall = aiResponse.tool_calls?.some(tc => 
    HITL_TOOLS.includes(tc.name)
  );
  
  // Check if any tool calls were made (HITL or otherwise)
  const hasAnyToolCall = aiResponse.tool_calls && aiResponse.tool_calls.length > 0;
  
  // Determine the new work state
  let newWorkState: AgentWorkState | null = null;
  
  if (isBriefComplete) {
    // Work complete - clear the state
    console.log("  [strategist] Work complete - clearing awaitingUserAction");
    newWorkState = null;
  } else if (nextPhase) {
    // Phase transition - update to new phase
    const newPhaseConfig = STRATEGIST_PHASES[nextPhase];
    console.log(`  [strategist] Entering phase: ${nextPhase}`);
    newWorkState = {
      agent: "strategist",
      phase: nextPhase,
      allowedTools: [...newPhaseConfig.allowedTools],
    };
  } else if (hasHITLToolCall) {
    // HITL tool call - stay in current phase, mark pending tool
    const hitlTool = aiResponse.tool_calls?.find(tc => HITL_TOOLS.includes(tc.name));
    console.log(`  [strategist] HITL tool call (${hitlTool?.name}) - awaiting user response`);
    newWorkState = {
      agent: "strategist",
      phase: currentPhase,
      pendingTool: hitlTool?.name,
      allowedTools: [...phaseConfig.allowedTools],
    };
  } else if (hasAnyToolCall) {
    // Non-HITL tool call - stay in current phase, continue working
    console.log(`  [strategist] Tool calls made - continuing in phase ${currentPhase}`);
    newWorkState = {
      agent: "strategist",
      phase: currentPhase,
      allowedTools: [...phaseConfig.allowedTools],
    };
  } else {
    // No tool calls, no phase transition - keep working
    console.log(`  [strategist] No tool calls - continuing in phase ${currentPhase}`);
    newWorkState = {
      agent: "strategist",
      phase: currentPhase,
      allowedTools: [...phaseConfig.allowedTools],
    };
  }

  // Build progress update for activeTask
  const progressUpdates: string[] = [];
  if (nextPhase === "searching_references") {
    progressUpdates.push("Strategist: Completed requirements gathering phase");
  }
  if (nextPhase === "creating_brief") {
    progressUpdates.push("Strategist: Completed framework/reference search");
  }
  if (isBriefComplete && parsedBrief) {
    progressUpdates.push(`Strategist: Created project brief for "${parsedBrief.purpose.substring(0, 50)}..."`);
  }

  // Update activeTask with progress if there are updates
  const activeTaskUpdate: Partial<ActiveTask> | null = progressUpdates.length > 0
    ? {
        progress: progressUpdates,
        // Clear assignment when work is complete
        ...(isBriefComplete && { assignedAgent: "orchestrator" as const }),
      }
    : null;

  return {
    messages: [response],
    currentAgent: "strategist",
    agentHistory: ["strategist"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
    // Include parsed project brief if available
    ...(parsedBrief && { projectBrief: parsedBrief }),
    // Update work state based on phase/HITL analysis
    awaitingUserAction: newWorkState,
    // Update activeTask with progress (reducer will merge with existing progress)
    ...(activeTaskUpdate && { activeTask: activeTaskUpdate as ActiveTask }),
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
    frameworks: input.frameworks,
    notes: input.notes,
  };
}

export default strategistNode;

