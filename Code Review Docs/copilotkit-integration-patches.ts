/**
 * CopilotKit v1.50 Integration Patches for orchestrator-agent.ts
 * 
 * This file contains the exact code additions needed to fix orphaned tool calls
 * in your multi-agent LangGraph + CopilotKit system.
 * 
 * Usage: Copy these sections into your orchestrator-agent.ts at the indicated locations
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { Command, END } from "@langchain/langgraph";
import { 
  copilotkitCustomizeConfig, 
  copilotkitEmitState 
} from "@copilotkit/sdk-js/langgraph";

// =============================================================================
// SECTION 1: TOOL CLASSIFICATION
// Add this after the TABLE_TOOLS_FOR_DATA_AGENT constant (~line 136)
// =============================================================================

/**
 * Gets the names of CopilotKit frontend actions from state.
 * These are actions defined on the frontend that CopilotKit can execute.
 */
function getCopilotKitActionNames(state: OrchestratorState): Set<string> {
  const actions = state.copilotkit?.actions ?? [];
  return new Set(actions.map((a: { name: string }) => a.name));
}

/**
 * Backend tools that are executed within LangGraph (not by CopilotKit).
 * These tools have their own handlers in sub-agent subgraphs.
 * 
 * NOTE: If you add new backend tools, add them here to prevent
 * them from being routed to the CopilotKit handler.
 */
const BACKEND_TOOL_NAMES = new Set([
  // Researcher backend tools (handled in researcher subgraph)
  "web_search",
  // Document tools that are handled as backend if you have backend implementations
  // Add any other backend-only tools here
]);

/**
 * Internal tools used by the supervisor for routing decisions.
 * These should never be emitted to CopilotKit.
 */
const INTERNAL_ROUTING_TOOLS = new Set([
  "supervisor_response",
]);

/**
 * Classifies tool calls by execution target.
 * 
 * @param toolCalls - Array of tool calls from the AI message
 * @param state - Current orchestrator state with CopilotKit context
 * @returns Object with backendToolCalls, frontendToolCalls, and routingToolCalls
 */
function classifyToolCalls(
  toolCalls: Array<{ name: string; id?: string; args: any }>,
  state: OrchestratorState
): {
  backendToolCalls: typeof toolCalls;
  frontendToolCalls: typeof toolCalls;
  routingToolCalls: typeof toolCalls;
} {
  const copilotKitActions = getCopilotKitActionNames(state);
  
  const backendToolCalls: typeof toolCalls = [];
  const frontendToolCalls: typeof toolCalls = [];
  const routingToolCalls: typeof toolCalls = [];
  
  for (const toolCall of toolCalls) {
    // Internal routing tools
    if (INTERNAL_ROUTING_TOOLS.has(toolCall.name)) {
      routingToolCalls.push(toolCall);
      continue;
    }
    
    // Known backend tools
    if (BACKEND_TOOL_NAMES.has(toolCall.name)) {
      backendToolCalls.push(toolCall);
      continue;
    }
    
    // CopilotKit frontend actions
    if (copilotKitActions.has(toolCall.name)) {
      frontendToolCalls.push(toolCall);
      continue;
    }
    
    // Unknown tools - treat as frontend (CopilotKit) by default
    // This is safer because:
    // 1. Most tools in your system are CopilotKit frontend actions
    // 2. If a frontend tool is misrouted to backend, it will fail silently
    // 3. If a backend tool is misrouted to frontend, CopilotKit will error visibly
    console.warn(`  [supervisor] Unknown tool "${toolCall.name}" - treating as frontend action`);
    frontendToolCalls.push(toolCall);
  }
  
  return { backendToolCalls, frontendToolCalls, routingToolCalls };
}


// =============================================================================
// SECTION 2: COPILOTKIT HANDLER NODE
// Add this after the helper functions section (~line 620)
// =============================================================================

/**
 * CopilotKit Handler Node
 * 
 * This node handles frontend tool calls by:
 * 1. Emitting them to CopilotKit for execution on the frontend
 * 2. Keeping the LangGraph run open (not routing to END)
 * 3. Routing back to supervisor to process tool results
 * 
 * WHY THIS EXISTS:
 * Without this node, when the supervisor calls a CopilotKit tool:
 * - The graph routes to END (run finishes)
 * - CopilotKit executes the tool on the frontend
 * - The tool result (ToolMessage) has nowhere to go
 * - Result only appears with the next user message (orphaned)
 * 
 * With this node:
 * - The graph routes HERE (run stays open)
 * - Tool calls are emitted to CopilotKit
 * - CopilotKit executes and returns results
 * - Results flow back to supervisor
 * - Supervisor processes results and responds
 * - THEN the run can end properly
 */
async function copilotKitHandlerNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Command> {
  console.log("\n[copilotkit-handler] ========================================");
  console.log("[copilotkit-handler] CopilotKit Tool Handler Node");
  console.log("[copilotkit-handler] ========================================");
  
  // Get the last message which should be an AI message with tool calls
  const messages = state.messages || [];
  
  if (messages.length === 0) {
    console.log("[copilotkit-handler] No messages in state - routing back to supervisor");
    return new Command({ goto: "supervisor" });
  }
  
  const lastMessage = messages[messages.length - 1];
  const msgType = (lastMessage as any)._getType?.() || (lastMessage as any).constructor?.name || "";
  
  // Verify we have an AI message
  if (!(msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk")) {
    console.log(`[copilotkit-handler] Last message is ${msgType}, not AI - routing back`);
    return new Command({ goto: "supervisor" });
  }
  
  const aiMessage = lastMessage as AIMessage;
  const toolCalls = aiMessage.tool_calls || [];
  
  if (toolCalls.length === 0) {
    console.log("[copilotkit-handler] AI message has no tool calls - routing back");
    return new Command({ goto: "supervisor" });
  }
  
  // Log what we're emitting
  console.log(`[copilotkit-handler] Emitting ${toolCalls.length} tool call(s) to CopilotKit:`);
  for (const tc of toolCalls) {
    console.log(`  - ${tc.name} (${tc.id?.slice(0, 8) || 'no-id'}...)`);
  }
  
  // Configure CopilotKit to emit these specific tool calls
  // Using specific tool names instead of `true` to be explicit
  const toolNames = toolCalls.map(tc => tc.name);
  
  try {
    const modifiedConfig = copilotkitCustomizeConfig(config, {
      emitToolCalls: toolNames,
      emitMessages: true,
    });
    
    // Emit current state to CopilotKit
    // This triggers the frontend to execute the tool calls
    await copilotkitEmitState(modifiedConfig, state);
    
    console.log("[copilotkit-handler] Tool calls emitted successfully");
  } catch (error) {
    console.error("[copilotkit-handler] Error emitting to CopilotKit:", error);
    // Even on error, route back to supervisor to handle gracefully
  }
  
  // CRITICAL: Route back to supervisor, NOT to END
  // The supervisor will either:
  // 1. Find tool results in state.messages and process them
  // 2. Detect pending tool calls via safety check and wait
  console.log("[copilotkit-handler] Routing back to supervisor to process results");
  
  return new Command({
    goto: "supervisor",
    update: {
      currentAgent: "orchestrator" as AgentType,
    },
  });
}


// =============================================================================
// SECTION 3: UPDATED SUPERVISOR ROUTING LOGIC
// Replace the existing routing logic for non-supervisor_response tool calls
// (around lines 991-1010)
// =============================================================================

// REPLACE THIS (old code):
/*
// CopilotKit frontend tool call (not supervisor_response) - route to END for CopilotKit to execute
console.log("  -> Routing to END (CopilotKit tool execution)");
return new Command({
  goto: END,
  update: {
    messages: updatedMessages,
    currentAgent: "orchestrator" as AgentType,
  },
});
*/

// WITH THIS (new code):
function handleNonRoutingToolCalls(
  aiResponse: AIMessage,
  state: OrchestratorState,
  updatedMessages: BaseMessage[]
): Command | null {
  // This function should be called when tool calls exist but no supervisor_response
  
  const toolCalls = aiResponse.tool_calls || [];
  if (toolCalls.length === 0) return null;
  
  // Classify the tool calls
  const { backendToolCalls, frontendToolCalls, routingToolCalls } = classifyToolCalls(
    toolCalls,
    state
  );
  
  console.log("  [routing] Tool classification results:");
  console.log(`    Routing tools: ${routingToolCalls.map(tc => tc.name).join(", ") || "none"}`);
  console.log(`    Backend tools: ${backendToolCalls.map(tc => tc.name).join(", ") || "none"}`);
  console.log(`    Frontend tools: ${frontendToolCalls.map(tc => tc.name).join(", ") || "none"}`);
  
  // If there are frontend tool calls, route to the CopilotKit handler
  if (frontendToolCalls.length > 0) {
    console.log("  -> Routing to copilotkit_handler (frontend tool execution)");
    return new Command({
      goto: "copilotkit_handler",
      update: {
        messages: updatedMessages,
        currentAgent: "orchestrator" as AgentType,
      },
    });
  }
  
  // If there are only backend tool calls, this is unusual from supervisor
  // Backend tools should be called from sub-agents with their own ToolNodes
  if (backendToolCalls.length > 0) {
    console.warn("  [routing] Backend tools called from supervisor - this is unusual");
    console.warn("  [routing] Backend tools should be called from specialized sub-agents");
    // Could route to a backend ToolNode if you add one to supervisor
    // For now, route to END and let the next invocation handle it
    return new Command({
      goto: END,
      update: {
        messages: updatedMessages,
        currentAgent: "orchestrator" as AgentType,
      },
    });
  }
  
  return null;
}


// =============================================================================
// SECTION 4: UPDATED GRAPH DEFINITION
// Modify the workflow definition to include the new handler
// =============================================================================

// UPDATE: Add "copilotkit_handler" to SUPERVISOR_ROUTING_DESTINATIONS
const SUPERVISOR_ROUTING_DESTINATIONS_UPDATED = [
  // Creative workflow agents
  "strategist",
  "researcher",
  "architect", 
  "writer",
  "visual_designer",
  "builder_agent",
  // Tool-specialized sub-agents
  "project_agent",
  "node_agent",
  "data_agent",
  "document_agent",
  "media_agent",
  "framework_agent",
  // NEW: CopilotKit handler for frontend tool execution
  "copilotkit_handler",
  // End (wait for user/CopilotKit)
  END,
];

// UPDATE: Add the handler node to the workflow
/*
const workflow = new StateGraph(OrchestratorStateAnnotation)
  // Supervisor node - uses Command for routing
  .addNode("supervisor", supervisorNode, { 
    ends: SUPERVISOR_ROUTING_DESTINATIONS_UPDATED  // Updated list
  })
  
  // NEW: CopilotKit handler for frontend tool calls
  .addNode("copilotkit_handler", copilotKitHandlerNode)
  
  // ... existing sub-agent subgraphs ...
  
  // Entry point -> supervisor
  .addEdge(START, "supervisor")
  
  // NEW: CopilotKit handler routes back to supervisor
  .addEdge("copilotkit_handler", "supervisor")
  
  // ... existing sub-agent edges ...
*/


// =============================================================================
// SECTION 5: INTEGRATION INTO supervisorNode
// Show where to integrate handleNonRoutingToolCalls in the supervisor
// =============================================================================

/*
INTEGRATION POINT:

In supervisorNode, after the supervisor_response handling but before the 
final "route to END" logic, add this check:

```typescript
// After handling supervisor_response routing (around line 968)

// Handle non-routing tool calls (CopilotKit frontend actions)
if (aiResponse.tool_calls && aiResponse.tool_calls.length > 0) {
  const hasRoutingTool = aiResponse.tool_calls.some(tc => 
    tc.name === "supervisor_response"
  );
  
  if (!hasRoutingTool) {
    // Use the classification-based routing
    const routingCommand = handleNonRoutingToolCalls(
      aiResponse, 
      state, 
      updatedMessages
    );
    
    if (routingCommand) {
      return routingCommand;
    }
  }
}

// Only reach here if no tool calls or all tools were routing tools
console.log("  -> Routing to END (waiting for user input)");
return new Command({
  goto: END,
  update: {
    messages: updatedMessages,
    currentAgent: "orchestrator" as AgentType,
  },
});
```
*/


// =============================================================================
// SECTION 6: SUB-AGENT COPILOTKIT CONFIG
// Add this pattern to sub-agents that call CopilotKit tools
// =============================================================================

/**
 * Example: How to update a sub-agent to properly emit CopilotKit tool calls
 * 
 * Add this to: architect.ts, writer.ts, node-agent.ts, etc.
 * Any agent that might call CopilotKit frontend actions.
 */
async function exampleSubAgentWithCopilotKitTools(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  // ... setup code ...
  
  // Get CopilotKit tools that this agent should have access to
  const frontendActions = state.copilotkit?.actions ?? [];
  const agentTools = frontendActions.filter((action: { name: string }) =>
    ["createNode", "updateNodeFields", "requestEditMode", /* etc */].includes(action.name)
  );
  
  // Bind tools to model
  const modelWithTools = agentModel.bindTools(agentTools);
  
  // IMPORTANT: Configure CopilotKit emission
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,  // or specific tool names
    emitMessages: true,
  });
  
  // Use the custom config when invoking
  const response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    customConfig  // <-- This ensures tool calls are emitted to CopilotKit
  );
  
  return {
    messages: [response],
    currentAgent: "example_agent",
    // ...
  };
}


// =============================================================================
// SECTION 7: DEBUG UTILITIES
// Add these helper functions for debugging tool call lifecycle
// =============================================================================

/**
 * Logs the current state of tool calls in the message history.
 * Useful for debugging orphaned tool call issues.
 */
function debugToolCallState(messages: BaseMessage[], prefix: string = ""): void {
  const toolCalls = new Map<string, {
    name: string;
    calledAt: number;
    hasResult: boolean;
    resultAt?: number;
  }>();
  
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    
    // Track tool calls from AI messages
    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = msg as AIMessage;
      for (const tc of aiMsg.tool_calls || []) {
        if (tc.id) {
          toolCalls.set(tc.id, {
            name: tc.name,
            calledAt: i,
            hasResult: false,
          });
        }
      }
    }
    
    // Track tool results
    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      const existing = toolCalls.get(toolMsg.tool_call_id);
      if (existing) {
        existing.hasResult = true;
        existing.resultAt = i;
      }
    }
  }
  
  console.log(`${prefix}=== TOOL CALL STATE ===`);
  if (toolCalls.size === 0) {
    console.log(`${prefix}  No tool calls in history`);
    return;
  }
  
  for (const [id, state] of toolCalls) {
    const status = state.hasResult 
      ? `✓ resolved at msg[${state.resultAt}]` 
      : `⏳ PENDING (called at msg[${state.calledAt}])`;
    console.log(`${prefix}  ${state.name} (${id.slice(0, 8)}...): ${status}`);
  }
  
  const pending = [...toolCalls.values()].filter(tc => !tc.hasResult);
  if (pending.length > 0) {
    console.log(`${prefix}  WARNING: ${pending.length} pending tool call(s)!`);
  }
}

/**
 * Validates that all tool calls have matching results.
 * Returns list of orphaned tool call IDs.
 */
function findOrphanedToolCalls(messages: BaseMessage[]): string[] {
  const toolCallIds = new Set<string>();
  const toolResultIds = new Set<string>();
  
  for (const msg of messages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    
    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = msg as AIMessage;
      for (const tc of aiMsg.tool_calls || []) {
        if (tc.id) toolCallIds.add(tc.id);
      }
    }
    
    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      toolResultIds.add(toolMsg.tool_call_id);
    }
  }
  
  // Find tool calls without results
  return [...toolCallIds].filter(id => !toolResultIds.has(id));
}


// =============================================================================
// SECTION 8: TYPE DEFINITIONS (if needed)
// =============================================================================

// Add to your state types if not already present
interface CopilotKitState {
  actions?: Array<{
    name: string;
    description?: string;
    parameters?: any[];
    available?: "remote" | "local" | "all";
  }>;
  context?: Record<string, any>;
}

// Ensure your OrchestratorState includes
interface OrchestratorState {
  // ... existing fields ...
  copilotkit?: CopilotKitState;
}


// =============================================================================
// EXPORTS (for use in orchestrator-agent.ts)
// =============================================================================

export {
  getCopilotKitActionNames,
  classifyToolCalls,
  copilotKitHandlerNode,
  handleNonRoutingToolCalls,
  debugToolCallState,
  findOrphanedToolCalls,
  BACKEND_TOOL_NAMES,
  INTERNAL_ROUTING_TOOLS,
  SUPERVISOR_ROUTING_DESTINATIONS_UPDATED,
};
