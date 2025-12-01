/**
 * Orchestrator Agent
 *
 * Coordinates workflow between specialized sub-agents for creating
 * impactful online training content.
 *
 * Sub-agents:
 * - Strategist: Discovers purpose, objectives, scope, constraints
 * - Researcher: Deep knowledge gathering on industry and topics
 * - Architect: Structures course for maximum learning impact
 * - Writer: Creates Level 6 content nodes
 * - Visual Designer: Defines aesthetics and writing tone
 *
 * Based on CopilotKit frontend actions pattern:
 * https://docs.copilotkit.ai/langgraph/frontend-actions
 */

import "dotenv/config";
import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { START, StateGraph, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";

// State and agent imports
import {
  OrchestratorStateAnnotation,
  OrchestratorState,
  AgentType,
  summarizeAgentContext,
  hasAgentOutput,
} from "./state/agent-state";

import {
  strategistNode,
  parseProjectBrief,
  researcherNode,
  researcherTools,
  parseResearchFindings,
  architectNode,
  parseCourseStructure,
  writerNode,
  extractContentOutput,
  visualDesignerNode,
  parseVisualDesign,
} from "./agents/index";

// ============================================================================
// MESSAGE FILTERING - Fix orphaned tool results and empty messages
// ============================================================================

function hasNonEmptyTextContent(msg: AIMessage): boolean {
  const content = msg.content;

  if (typeof content === "string") {
    return content.trim().length > 0;
  }

  if (Array.isArray(content)) {
    for (const block of content) {
      if (typeof block === "string" && block.trim().length > 0) {
        return true;
      }
      if (typeof block === "object" && block !== null) {
        if ("type" in block && block.type === "text" && "text" in block) {
          const text = (block as any).text;
          if (typeof text === "string" && text.trim().length > 0) {
            return true;
          }
        }
      }
    }
  }

  if (msg.tool_calls && msg.tool_calls.length > 0) {
    return true;
  }

  return false;
}

function filterOrphanedToolResults(messages: BaseMessage[]): BaseMessage[] {
  const filtered: BaseMessage[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";

    if (msgType === "ai" || msgType === "AIMessage") {
      const aiMsg = msg as AIMessage;
      if (!hasNonEmptyTextContent(aiMsg)) {
        console.log(`  [FILTER] Removing AI message with empty text content`);
        continue;
      }
    }

    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      const toolCallId = toolMsg.tool_call_id;

      let hasMatchingToolUse = false;
      for (let j = filtered.length - 1; j >= 0; j--) {
        const prevMsg = filtered[j];
        const prevType = (prevMsg as any)._getType?.() || (prevMsg as any).constructor?.name || "";

        if (prevType === "ai" || prevType === "AIMessage") {
          const aiMsg = prevMsg as AIMessage;
          if (aiMsg.tool_calls?.some((tc) => tc.id === toolCallId)) {
            hasMatchingToolUse = true;
          }
          break;
        }
      }

      if (!hasMatchingToolUse) {
        console.log(`  [FILTER] Removing orphaned tool result: ${toolCallId}`);
        continue;
      }
    }

    filtered.push(msg);
  }

  return filtered;
}

function trimMessages(messages: BaseMessage[], keepRecent: number = 40): BaseMessage[] {
  if (messages.length <= keepRecent) {
    return messages;
  }

  const systemMessages = messages.filter((m) => {
    const msgType = (m as any)._getType?.() || (m as any).constructor?.name || "";
    return msgType === "system" || msgType === "SystemMessage";
  });

  const otherMessages = messages.filter((m) => {
    const msgType = (m as any)._getType?.() || (m as any).constructor?.name || "";
    return msgType !== "system" && msgType !== "SystemMessage";
  });

  let startIdx = Math.max(0, otherMessages.length - keepRecent);

  while (startIdx < otherMessages.length) {
    const msg = otherMessages[startIdx];
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";

    if (msgType === "human" || msgType === "HumanMessage") {
      break;
    }

    if (msgType === "tool" || msgType === "ToolMessage") {
      startIdx++;
      continue;
    }

    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = msg as any;
      const hasToolCalls = aiMsg.tool_calls?.length > 0;
      if (!hasToolCalls) {
        break;
      }
      startIdx++;
      continue;
    }

    startIdx++;
  }

  const recentMessages = otherMessages.slice(startIdx);

  console.log(
    `  [TRIM] Messages: ${messages.length} -> ${systemMessages.length + recentMessages.length}`
  );

  return [...systemMessages, ...recentMessages];
}

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const orchestratorModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 32000,
  thinking: {
    type: "enabled",
    budget_tokens: 10000,
  },
  temperature: 1,
});

// ============================================================================
// ORCHESTRATOR SYSTEM PROMPT
// ============================================================================

const ORCHESTRATOR_SYSTEM_PROMPT = `You are The Orchestrator - the coordinator for a multi-agent system that creates impactful online training content.

## Your Team

You have 5 specialized sub-agents, each with specific capabilities:

### 1. The Strategist
**When to use**: First step for new projects, or when goals/scope are unclear
**Capabilities**: Asks clarifying questions, gathers requirements, defines project brief
**Output**: Project brief with purpose, objectives, scope, constraints

### 2. The Researcher
**When to use**: After strategy is defined, to gather deep knowledge
**Capabilities**: Web search, document search, RAG queries
**Output**: Research findings with industry context, key topics, regulations

### 3. The Architect
**When to use**: After research is complete, to design course structure
**Capabilities**: Analyzes hierarchy, designs node structure, plans content flow
**Output**: Course structure with planned nodes and learning progression

### 4. The Writer
**When to use**: After structure is approved, to create actual content
**Capabilities**: Creates nodes, writes content, manages edit mode
**Output**: Written content nodes following the structure

### 5. The Visual Designer
**When to use**: Early in process to define look/feel, or when user asks about design
**Capabilities**: Presents design options, defines colors, fonts, tone
**Output**: Visual design specification for the course

## Workflow Patterns

### Full Course Creation
1. Strategist → Gather requirements
2. Visual Designer → Define aesthetics (can run parallel with research)
3. Researcher → Deep knowledge gathering
4. Architect → Design course structure
5. Get approval → User reviews structure
6. Writer → Create content nodes

### Quick Content Addition
1. Architect → Plan new content location
2. Writer → Create the content

### Research Deep-Dive
1. Researcher → Extended research on specific topics

### Design Refresh
1. Visual Designer → New design options

## Your Role

As the Orchestrator, you:
1. **Analyze requests** - Understand what the user needs
2. **Route to agents** - Delegate to the right specialist
3. **Manage context** - Pass relevant information between agents
4. **Coordinate workflow** - Ensure proper sequencing
5. **Handle general tasks** - Answer questions, provide status updates

## Your Tools

You have access to:
- **requestPlanApproval** - Get user approval before major actions
- **requestActionApproval** - Confirm sensitive operations
- **offerOptions** - Present choices when needed
- **getProjectHierarchyInfo** - Understand project structure
- **getNodeChildren** - Check existing content

## Decision Making

When a user message arrives, decide:

1. **Can I handle this directly?** - Simple questions, status updates, clarifications
2. **Which agent should handle this?** - Complex tasks need specialists
3. **What context do they need?** - Pass relevant brief/research/structure
4. **What's the optimal sequence?** - Some agents depend on others' output

## Guidelines

- Always start new projects with the Strategist
- Research before Architecture
- Get approval before Writing begins
- Visual Design can run early in parallel
- Keep the user informed of progress
- Summarize agent outputs for the user
- Don't call multiple agents in one turn - process sequentially
- If unsure, ask the user for clarification

## Current Context Summary

This section will be updated with the current state of agent outputs.`;

// ============================================================================
// ORCHESTRATOR NODE
// ============================================================================

async function orchestratorNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[orchestrator] ============ Orchestrator Agent ============");
  console.log("  Agent context:", summarizeAgentContext(state));
  console.log("  Messages count:", state.messages?.length ?? 0);
  console.log("  Current agent:", state.currentAgent);
  console.log("  Agent history:", state.agentHistory?.join(" -> ") || "none");

  // Get frontend tools for the orchestrator
  const frontendActions = state.copilotkit?.actions ?? [];
  const orchestratorTools = frontendActions.filter((action: { name: string }) =>
    [
      "requestPlanApproval",
      "requestActionApproval",
      "offerOptions",
      "getProjectHierarchyInfo",
      "getNodeChildren",
    ].includes(action.name)
  );

  console.log("  Available tools:", orchestratorTools.map((t: { name: string }) => t.name).join(", ") || "none");

  // Build dynamic system prompt with context
  let systemContent = ORCHESTRATOR_SYSTEM_PROMPT;

  // Add current context summary
  systemContent += `\n\n## Current Agent Context\n`;

  if (state.projectBrief) {
    systemContent += `\n### Project Brief (from Strategist) ✓
- Purpose: ${state.projectBrief.purpose}
- Industry: ${state.projectBrief.industry}
- Target Audience: ${state.projectBrief.targetAudience}
- Objectives: ${state.projectBrief.objectives.length} defined`;
  } else {
    systemContent += `\n### Project Brief: Not yet defined
Consider starting with the Strategist to gather requirements.`;
  }

  if (state.researchFindings) {
    systemContent += `\n\n### Research Findings (from Researcher) ✓
- Key Topics: ${state.researchFindings.keyTopics.length}
- Regulations: ${state.researchFindings.regulations.length}
- Citations: ${state.researchFindings.citations.length}`;
  } else {
    systemContent += `\n\n### Research: Not yet conducted`;
  }

  if (state.courseStructure) {
    systemContent += `\n\n### Course Structure (from Architect) ✓
- Total Nodes: ${state.courseStructure.totalNodes}
- Max Depth: ${state.courseStructure.maxDepth}
- Summary: ${state.courseStructure.summary}`;
  } else {
    systemContent += `\n\n### Course Structure: Not yet designed`;
  }

  if (state.visualDesign) {
    systemContent += `\n\n### Visual Design (from Visual Designer) ✓
- Theme: ${state.visualDesign.theme}
- Tone: ${state.visualDesign.writingTone.tone}
- Style: ${state.visualDesign.typography.style}`;
  } else {
    systemContent += `\n\n### Visual Design: Not yet defined`;
  }

  if (state.writtenContent && state.writtenContent.length > 0) {
    systemContent += `\n\n### Written Content (from Writer) ✓
- Nodes Created: ${state.writtenContent.length}`;
  }

  // Add routing instructions
  systemContent += `\n\n## Routing Instructions

To delegate to a sub-agent, respond with a clear routing directive in your message:
- "I'll have the Strategist gather requirements..."
- "Let me ask the Researcher to look into..."
- "The Architect should design..."
- "The Writer will create..."
- "Let's have the Visual Designer propose..."

The system will detect these and route accordingly.`;

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = orchestratorTools.length > 0
    ? orchestratorModel.bindTools(orchestratorTools)
    : orchestratorModel;

  // Prepare messages
  const filteredMessages = filterOrphanedToolResults(state.messages || []);
  const trimmedMessages = trimMessages(filteredMessages, 40);

  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
  });

  console.log("  Invoking orchestrator model...");

  const response = await modelWithTools.invoke(
    [systemMessage, ...trimmedMessages],
    customConfig
  );

  console.log("  Orchestrator response received");

  // Log thinking and response
  const aiResponse = response as AIMessage;
  if (aiResponse.content && Array.isArray(aiResponse.content)) {
    for (const block of aiResponse.content) {
      if (typeof block === "object" && block !== null) {
        if ("type" in block && block.type === "thinking") {
          console.log("\n  [THINKING] ================================");
          const thinking = ((block as any).thinking || "").substring(0, 500);
          console.log("  " + thinking + (thinking.length >= 500 ? "..." : ""));
          console.log("  [/THINKING] ===============================\n");
        }
      }
    }
  }

  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // Detect routing directives in response
  const responseText = typeof aiResponse.content === "string"
    ? aiResponse.content
    : Array.isArray(aiResponse.content)
    ? aiResponse.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";

  const routingDecision = detectRoutingDirective(responseText);
  if (routingDecision) {
    console.log(`  Detected routing to: ${routingDecision.nextAgent}`);
  }

  return {
    messages: [response],
    currentAgent: "orchestrator",
    routingDecision,
  };
}

// ============================================================================
// ROUTING DETECTION
// ============================================================================

function detectRoutingDirective(text: string): { nextAgent: AgentType; reason: string; task: string } | null {
  const lower = text.toLowerCase();

  if (lower.includes("strategist") && (lower.includes("gather") || lower.includes("ask") || lower.includes("discover") || lower.includes("clarify"))) {
    return { nextAgent: "strategist", reason: "Strategy/requirements gathering needed", task: text };
  }

  if (lower.includes("researcher") && (lower.includes("research") || lower.includes("look into") || lower.includes("investigate") || lower.includes("search"))) {
    return { nextAgent: "researcher", reason: "Research/knowledge gathering needed", task: text };
  }

  if (lower.includes("architect") && (lower.includes("design") || lower.includes("structure") || lower.includes("plan") || lower.includes("organize"))) {
    return { nextAgent: "architect", reason: "Course structure design needed", task: text };
  }

  if (lower.includes("writer") && (lower.includes("write") || lower.includes("create") || lower.includes("content"))) {
    return { nextAgent: "writer", reason: "Content creation needed", task: text };
  }

  if (lower.includes("visual designer") && (lower.includes("design") || lower.includes("style") || lower.includes("theme") || lower.includes("color"))) {
    return { nextAgent: "visual_designer", reason: "Visual design needed", task: text };
  }

  return null;
}

// ============================================================================
// ROUTING LOGIC
// ============================================================================

type NodeName = "orchestrator" | "strategist" | "researcher" | "architect" | "writer" | "visual_designer" | "execute_backend_tools" | "__end__";

function routeFromOrchestrator(state: OrchestratorState): NodeName {
  console.log("\n[routing] From orchestrator...");

  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // Check for tool calls
  if (lastMessage.tool_calls?.length) {
    const toolNames = lastMessage.tool_calls.map((tc) => tc.name);
    console.log("  Tool calls detected:", toolNames.join(", "));

    // Check if any are backend tools (researcher's tools)
    const backendToolNames = new Set(researcherTools.map((t) => t.name));
    const hasBackendTool = toolNames.some((name) => backendToolNames.has(name));

    if (hasBackendTool) {
      console.log("  -> Route to execute_backend_tools");
      return "execute_backend_tools";
    }

    // Frontend tools route to END for CopilotKit execution
    console.log("  -> Route to __end__ (frontend tools)");
    return "__end__";
  }

  // Check for routing decision
  if (state.routingDecision) {
    const nextAgent = state.routingDecision.nextAgent;
    console.log(`  Routing decision: ${nextAgent}`);

    if (nextAgent !== "orchestrator") {
      return nextAgent as NodeName;
    }
  }

  // No tool calls and no routing - done
  console.log("  -> Route to __end__ (no action needed)");
  return "__end__";
}

function routeFromSubAgent(state: OrchestratorState): NodeName {
  console.log("\n[routing] From sub-agent...");

  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // Check for tool calls
  if (lastMessage.tool_calls?.length) {
    const toolNames = lastMessage.tool_calls.map((tc) => tc.name);
    console.log("  Tool calls detected:", toolNames.join(", "));

    // Backend tools need execution
    const backendToolNames = new Set(researcherTools.map((t) => t.name));
    const hasBackendTool = toolNames.some((name) => backendToolNames.has(name));

    if (hasBackendTool) {
      console.log("  -> Route to execute_backend_tools");
      return "execute_backend_tools";
    }

    // Frontend tools route to END
    console.log("  -> Route to __end__ (frontend tools)");
    return "__end__";
  }

  // No tool calls - return to orchestrator
  console.log("  -> Route to orchestrator");
  return "orchestrator";
}

function routeAfterToolExecution(state: OrchestratorState): NodeName {
  console.log("\n[routing] After tool execution...");

  // Return to the current agent to process results
  const currentAgent = state.currentAgent || "orchestrator";
  console.log(`  -> Route back to ${currentAgent}`);
  return currentAgent as NodeName;
}

// ============================================================================
// TOOL EXECUTION NODE
// ============================================================================

const backendToolNode = new ToolNode(researcherTools);

// ============================================================================
// GRAPH DEFINITION
// ============================================================================

console.log("[orchestrator] Building workflow graph...");

const workflow = new StateGraph(OrchestratorStateAnnotation)
  // Add all nodes
  .addNode("orchestrator", orchestratorNode)
  .addNode("strategist", strategistNode)
  .addNode("researcher", researcherNode)
  .addNode("architect", architectNode)
  .addNode("writer", writerNode)
  .addNode("visual_designer", visualDesignerNode)
  .addNode("execute_backend_tools", backendToolNode)

  // Entry point
  .addEdge(START, "orchestrator")

  // Orchestrator routing
  .addConditionalEdges("orchestrator", routeFromOrchestrator, {
    strategist: "strategist",
    researcher: "researcher",
    architect: "architect",
    writer: "writer",
    visual_designer: "visual_designer",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })

  // Sub-agent routing (all go back to orchestrator or to tools)
  .addConditionalEdges("strategist", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("researcher", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("architect", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("writer", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })
  .addConditionalEdges("visual_designer", routeFromSubAgent, {
    orchestrator: "orchestrator",
    execute_backend_tools: "execute_backend_tools",
    __end__: END,
  })

  // After tool execution, route back to current agent
  .addConditionalEdges("execute_backend_tools", routeAfterToolExecution, {
    orchestrator: "orchestrator",
    strategist: "strategist",
    researcher: "researcher",
    architect: "architect",
    writer: "writer",
    visual_designer: "visual_designer",
  });

// ============================================================================
// PERSISTENCE SETUP
// ============================================================================

const SUPABASE_DB_URL = process.env.SUPABASE_DB_URL || "postgresql://postgres:postgres@localhost:15322/postgres";

console.log("[orchestrator] Initializing PostgreSQL checkpointer...");
console.log("[orchestrator] DB URL:", SUPABASE_DB_URL.replace(/:[^:@]+@/, ":****@"));

const checkpointer = PostgresSaver.fromConnString(SUPABASE_DB_URL);

await checkpointer.setup();
console.log("[orchestrator] PostgreSQL checkpointer initialized successfully");

// Compile the graph
export const agent = workflow.compile({
  checkpointer,
});

console.log("[orchestrator] Workflow graph compiled successfully");
console.log("[orchestrator] Nodes: orchestrator, strategist, researcher, architect, writer, visual_designer, execute_backend_tools");

