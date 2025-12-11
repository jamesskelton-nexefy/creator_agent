# CopilotKit v1.50 + LangGraph 1.0.4 Complete Upgrade Plan

## Problem Summary

The current architecture has a critical flaw: **when the supervisor calls a CopilotKit frontend tool, the graph routes to `END` prematurely**. This causes:

- CopilotKit executes the tool on the frontend
- The `ToolMessage` result has nowhere to go (orphaned)
- Results only appear when the next user message arrives
- Multi-step workflows break (e.g., architect's `requestPlanApproval`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CURRENT FLOW (BROKEN)                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Message → Supervisor calls createNode → Routes to END (run finishes)  │
│                                                     ↓                        │
│                                        CopilotKit executes tool              │
│                                                     ↓                        │
│                                        ToolMessage ORPHANED! ❌              │
│                                                     ↓                        │
│                                   User must send another message...          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FIXED FLOW                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Message → Supervisor calls createNode → Routes to copilotkit_handler  │
│                                                     ↓                        │
│                                     Emits tool call, keeps run OPEN          │
│                                                     ↓                        │
│                                        CopilotKit executes tool              │
│                                                     ↓                        │
│                                     Result flows back to supervisor ✅       │
│                                                     ↓                        │
│                                        Supervisor responds                   │
│                                                     ↓                        │
│                                          Run ends properly                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Version Compatibility Matrix

| Package | Current | Target | Notes |
|---------|---------|--------|-------|
| @langchain/langgraph | ? | ^1.0.4 | `ends` param required for Command routing |
| @langchain/core | ? | ^1.1.4 | Compatible with LangGraph 1.0.4 |
| @langchain/openai | ? | ^0.5.0 | Tool calling support |
| @copilotkit/sdk-js | ? | ^1.50.0 | Backend SDK |
| @copilotkit/runtime | ? | ^1.50.0 | Runtime |
| @copilotkit/react-core | ? | ^1.50.0 | Frontend hooks |
| @copilotkit/react-ui | ? | ^1.50.0 | UI components |

---

## Files to Modify

### Backend
| File | Changes |
|------|---------|
| `deepagentsjs/package.json` | Update CopilotKit + LangGraph versions |
| `deepagentsjs/examples/research/orchestrator-agent.ts` | Tool classification, handler node, routing |
| `deepagentsjs/examples/research/agents/strategist.ts` | Add copilotkitCustomizeConfig |
| `deepagentsjs/examples/research/agents/architect.ts` | Add copilotkitCustomizeConfig (**Critical**) |
| `deepagentsjs/examples/research/agents/writer.ts` | Add copilotkitCustomizeConfig |
| `deepagentsjs/examples/research/agents/visual-designer.ts` | Add copilotkitCustomizeConfig |
| `deepagentsjs/examples/research/agents/builder-agent.ts` | Add copilotkitCustomizeConfig (**Critical**) |
| `deepagentsjs/examples/research/agents/project-agent.ts` | Add copilotkitCustomizeConfig (**Critical**) |
| `deepagentsjs/examples/research/agents/node-agent.ts` | Add copilotkitCustomizeConfig (**Critical**) |
| `deepagentsjs/examples/research/agents/data-agent.ts` | Add copilotkitCustomizeConfig |
| `deepagentsjs/examples/research/agents/document-agent.ts` | Add copilotkitCustomizeConfig |
| `deepagentsjs/examples/research/agents/media-agent.ts` | Add copilotkitCustomizeConfig |
| `deepagentsjs/examples/research/agents/framework-agent.ts` | Add copilotkitCustomizeConfig |

### Frontend
| File | Changes |
|------|---------|
| `frontend/package.json` | Update CopilotKit packages |
| `frontend/src/App.tsx` | Thread persistence, provider config |
| `frontend/src/hooks/useCourseActions.ts` (or equivalent) | Add `available: "remote"` |
| `frontend/src/components/ToolCallDebugger.tsx` | New debug component |

---

## Phase 1: Package Updates

### 1.1 Backend - `deepagentsjs/package.json`

```json
{
  "dependencies": {
    "@langchain/langgraph": "^1.0.4",
    "@langchain/core": "^1.1.4",
    "@langchain/openai": "^0.5.0",
    "@copilotkit/sdk-js": "^1.50.0",
    "@copilotkit/runtime": "^1.50.0"
  }
}
```

### 1.2 Frontend - `frontend/package.json`

```json
{
  "dependencies": {
    "@copilotkit/react-core": "^1.50.0",
    "@copilotkit/react-ui": "^1.50.0"
  }
}
```

---

## Phase 2: Backend - Tool Classification System

### 2.1 Add Constants and Classification Function

**Location:** After `TABLE_TOOLS_FOR_DATA_AGENT` constant (~line 136) in `orchestrator-agent.ts`

```typescript
// =============================================================================
// TOOL CLASSIFICATION SYSTEM
// =============================================================================

/**
 * Backend tools - executed within LangGraph via ToolNode.
 * These tools have handlers in sub-agent subgraphs or the main graph.
 * 
 * IMPORTANT: Add any new backend tools here to prevent them from being
 * misrouted to the CopilotKit handler.
 */
const BACKEND_TOOL_NAMES = new Set([
  // Researcher subgraph tools
  "web_search",
  "tavily_search",
  
  // Add other backend-only tools as discovered
  // Example: "database_query", "file_read", etc.
]);

/**
 * Internal routing tools - used by supervisor for agent delegation.
 * These should NEVER be emitted to CopilotKit.
 */
const INTERNAL_ROUTING_TOOLS = new Set([
  "supervisor_response",
]);

/**
 * Gets the names of CopilotKit frontend actions from state.
 * These actions are defined on the frontend and executed by CopilotKit.
 */
function getCopilotKitActionNames(state: OrchestratorState): Set<string> {
  const actions = state.copilotkit?.actions ?? [];
  return new Set(actions.map((a: { name: string }) => a.name));
}

/**
 * Tool call classification result.
 */
interface ClassifiedToolCalls {
  backendToolCalls: Array<{ name: string; id?: string; args: any }>;
  frontendToolCalls: Array<{ name: string; id?: string; args: any }>;
  routingToolCalls: Array<{ name: string; id?: string; args: any }>;
}

/**
 * Classifies tool calls by their execution target.
 * 
 * Classification priority:
 * 1. Internal routing tools (supervisor_response) → routingToolCalls
 * 2. Known backend tools (web_search, etc.) → backendToolCalls
 * 3. CopilotKit frontend actions → frontendToolCalls
 * 4. Unknown tools → frontendToolCalls (safer default)
 * 
 * @param toolCalls - Array of tool calls from the AI message
 * @param state - Current orchestrator state with CopilotKit context
 * @returns Classified tool calls by execution target
 */
function classifyToolCalls(
  toolCalls: Array<{ name: string; id?: string; args: any }>,
  state: OrchestratorState
): ClassifiedToolCalls {
  const copilotKitActions = getCopilotKitActionNames(state);
  
  const backendToolCalls: ClassifiedToolCalls["backendToolCalls"] = [];
  const frontendToolCalls: ClassifiedToolCalls["frontendToolCalls"] = [];
  const routingToolCalls: ClassifiedToolCalls["routingToolCalls"] = [];
  
  for (const toolCall of toolCalls) {
    // 1. Internal routing tools
    if (INTERNAL_ROUTING_TOOLS.has(toolCall.name)) {
      routingToolCalls.push(toolCall);
      continue;
    }
    
    // 2. Known backend tools
    if (BACKEND_TOOL_NAMES.has(toolCall.name)) {
      backendToolCalls.push(toolCall);
      continue;
    }
    
    // 3. CopilotKit frontend actions
    if (copilotKitActions.has(toolCall.name)) {
      frontendToolCalls.push(toolCall);
      continue;
    }
    
    // 4. Unknown tools - default to frontend (safer)
    // Rationale: 
    // - Most tools in this system are CopilotKit actions
    // - Frontend misroute → visible error in CopilotKit
    // - Backend misroute → silent failure
    console.warn(`  [classifyToolCalls] Unknown tool "${toolCall.name}" - treating as frontend action`);
    frontendToolCalls.push(toolCall);
  }
  
  // Debug logging
  if (toolCalls.length > 0) {
    console.log("  [classifyToolCalls] Classification result:");
    console.log(`    - Routing: ${routingToolCalls.map(t => t.name).join(", ") || "none"}`);
    console.log(`    - Backend: ${backendToolCalls.map(t => t.name).join(", ") || "none"}`);
    console.log(`    - Frontend: ${frontendToolCalls.map(t => t.name).join(", ") || "none"}`);
  }
  
  return { backendToolCalls, frontendToolCalls, routingToolCalls };
}
```

---

## Phase 3: Backend - CopilotKit Handler Node

### 3.1 Add Handler Node

**Location:** After helper functions (~line 620) in `orchestrator-agent.ts`

```typescript
// =============================================================================
// COPILOTKIT HANDLER NODE
// =============================================================================

import {
  copilotkitCustomizeConfig,
  copilotkitEmitState,
} from "@copilotkit/sdk-js/langgraph";

/**
 * CopilotKit Handler Node
 * 
 * This node handles frontend tool calls by:
 * 1. Emitting tool calls to CopilotKit via AG-UI protocol
 * 2. Keeping the graph run OPEN (routing back to supervisor)
 * 3. Allowing the supervisor to receive tool results
 * 
 * CRITICAL: This node routes to "supervisor", NOT to END.
 * Routing to END would cause orphaned tool calls.
 */
async function copilotKitHandlerNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Command> {
  console.log("\n========================================");
  console.log("[copilotkit_handler] Processing frontend tool calls");
  console.log("========================================");
  
  try {
    // Get pending frontend actions (if stored in state)
    const pendingActions = state.pendingFrontendActions ?? [];
    console.log(`  Pending frontend actions: ${pendingActions.length}`);
    
    // Configure CopilotKit to emit tool calls
    const modifiedConfig = copilotkitCustomizeConfig(config, {
      emitToolCalls: true,   // Emit tool calls to frontend
      emitMessages: true,    // Emit message updates
    });
    
    // Emit current state to CopilotKit
    // This triggers the frontend to execute the pending tool calls
    await copilotkitEmitState(modifiedConfig, state);
    
    console.log("  ✓ Tool calls emitted to CopilotKit");
    console.log("  → Routing back to supervisor to await results");
    
    // CRITICAL: Route back to supervisor, NOT to END
    // The supervisor's safety check will wait for tool results
    return new Command({
      goto: "supervisor",
      update: {
        pendingFrontendActions: [], // Clear pending actions
        currentAgent: "orchestrator" as AgentType,
      },
    });
    
  } catch (error) {
    console.error("  [copilotkit_handler] Error:", error);
    
    // On error, still route to supervisor with error state
    // Let supervisor handle the error gracefully
    return new Command({
      goto: "supervisor",
      update: {
        pendingFrontendActions: [],
        currentAgent: "orchestrator" as AgentType,
        lastError: {
          node: "copilotkit_handler",
          message: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
      },
    });
  }
}
```

---

## Phase 4: Backend - Update Supervisor Routing Logic

### 4.1 Update Routing After LLM Response

**Location:** Replace lines 991-999 in `orchestrator-agent.ts` (the "route to END for CopilotKit" section)

```typescript
// =============================================================================
// UPDATED ROUTING LOGIC - Handles frontend tools properly
// =============================================================================

// Get tool calls from AI response
const toolCalls = aiResponse.tool_calls ?? [];

if (toolCalls.length > 0) {
  // Classify all tool calls
  const { backendToolCalls, frontendToolCalls, routingToolCalls } = classifyToolCalls(
    toolCalls,
    state
  );
  
  // Handle routing tool (supervisor_response)
  if (routingToolCalls.length > 0) {
    const routingCall = routingToolCalls[0];
    const targetAgent = routingCall.args?.next;
    
    if (targetAgent && SUPERVISOR_ROUTING_DESTINATIONS.includes(targetAgent)) {
      console.log(`  -> Routing to agent: ${targetAgent}`);
      return new Command({
        goto: targetAgent,
        update: {
          messages: updatedMessages,
          currentAgent: targetAgent as AgentType,
          pendingFrontendActions: frontendToolCalls, // Preserve any frontend calls
        },
      });
    }
  }
  
  // Handle mixed calls: backend + frontend
  // Execute backend tools FIRST, then handle frontend
  if (backendToolCalls.length > 0 && frontendToolCalls.length > 0) {
    console.log("  -> Mixed tool calls detected");
    console.log(`     Backend: ${backendToolCalls.map(t => t.name).join(", ")}`);
    console.log(`     Frontend: ${frontendToolCalls.map(t => t.name).join(", ")}`);
    console.log("  -> Routing to toolNode first, frontend calls stored");
    
    return new Command({
      goto: "toolNode", // Process backend tools first
      update: {
        messages: updatedMessages,
        currentAgent: "orchestrator" as AgentType,
        pendingFrontendActions: frontendToolCalls, // Store for later
      },
    });
  }
  
  // Handle backend-only calls
  if (backendToolCalls.length > 0 && frontendToolCalls.length === 0) {
    console.log(`  -> Backend tools only: ${backendToolCalls.map(t => t.name).join(", ")}`);
    console.log("  -> Routing to toolNode");
    
    return new Command({
      goto: "toolNode",
      update: {
        messages: updatedMessages,
        currentAgent: "orchestrator" as AgentType,
      },
    });
  }
  
  // Handle frontend-only calls (CopilotKit actions)
  if (frontendToolCalls.length > 0 && backendToolCalls.length === 0) {
    console.log(`  -> Frontend tools only: ${frontendToolCalls.map(t => t.name).join(", ")}`);
    console.log("  -> Routing to copilotkit_handler");
    
    // CRITICAL: Route to handler, NOT to END
    return new Command({
      goto: "copilotkit_handler",
      update: {
        messages: updatedMessages,
        currentAgent: "orchestrator" as AgentType,
        pendingFrontendActions: frontendToolCalls,
      },
    });
  }
}

// No tool calls - check if we should end
if (!toolCalls.length) {
  // ... existing logic for final response ...
}
```

### 4.2 Update Safety Check

**Location:** Lines 646-678 (the existing safety check)

Update to route to handler instead of END:

```typescript
// =============================================================================
// SAFETY CHECK - Updated to use handler
// =============================================================================

if (unresolvedCalls.length > 0) {
  console.log(`  [SAFETY] Last message has ${unresolvedCalls.length} unresolved tool_calls`);
  console.log(`  [SAFETY] Tool IDs: ${unresolvedCalls.map(tc => tc.id).join(", ")}`);
  
  // Check if these are frontend or backend tools
  const { frontendToolCalls, backendToolCalls } = classifyToolCalls(unresolvedCalls, state);
  
  if (frontendToolCalls.length > 0) {
    console.log("  [SAFETY] Unresolved frontend tools - routing to handler");
    // Route to handler to properly emit and await results
    return new Command({
      goto: "copilotkit_handler",
      update: {
        currentAgent: "orchestrator" as AgentType,
        pendingFrontendActions: frontendToolCalls,
      },
    });
  }
  
  if (backendToolCalls.length > 0) {
    console.log("  [SAFETY] Unresolved backend tools - routing to toolNode");
    return new Command({
      goto: "toolNode",
      update: { currentAgent: "orchestrator" as AgentType },
    });
  }
  
  // Fallback: wait for results
  console.log("  [SAFETY] Waiting for tool results - skipping LLM invocation");
  return new Command({
    goto: END,
    update: { currentAgent: "orchestrator" as AgentType },
  });
}
```

---

## Phase 5: Backend - Update Graph Definition

### 5.1 Update Routing Destinations

**Location:** Find `SUPERVISOR_ROUTING_DESTINATIONS` constant

```typescript
const SUPERVISOR_ROUTING_DESTINATIONS = [
  // Sub-agents
  "strategist",
  "researcher", 
  "architect",
  "writer",
  "visual_designer",
  "builder_agent",
  "project_agent",
  "node_agent",
  "data_agent",
  "document_agent",
  "media_agent",
  "framework_agent",
  
  // Tool execution nodes
  "toolNode",           // Backend tools
  "copilotkit_handler", // Frontend tools (NEW)
  
  // Terminal
  END,
];
```

### 5.2 Update StateGraph Definition

**Location:** Find the `new StateGraph(...)` section

```typescript
// =============================================================================
// GRAPH DEFINITION - LangGraph 1.0.4 Compatible
// =============================================================================

// Add pendingFrontendActions to state annotation if not present
const OrchestratorStateAnnotation = Annotation.Root({
  // ... existing fields ...
  
  pendingFrontendActions: Annotation<Array<{ name: string; id?: string; args: any }>>({
    reducer: (_, update) => update,
    default: () => [],
  }),
  
  lastError: Annotation<{ node: string; message: string; timestamp: string } | null>({
    reducer: (_, update) => update,
    default: () => null,
  }),
});

const workflow = new StateGraph(OrchestratorStateAnnotation)
  // Supervisor node with Command routing
  // CRITICAL (LangGraph 1.0.4): `ends` parameter is REQUIRED when node returns Command
  .addNode("supervisor", supervisorNode, {
    ends: SUPERVISOR_ROUTING_DESTINATIONS,
  })
  
  // CopilotKit handler for frontend tool calls (NEW)
  .addNode("copilotkit_handler", copilotKitHandlerNode)
  
  // Backend tool execution
  .addNode("toolNode", toolNode)
  
  // Sub-agents
  .addNode("strategist", strategistNode)
  .addNode("researcher", researcherSubgraph)
  .addNode("architect", architectNode)
  .addNode("writer", writerNode)
  .addNode("visual_designer", visualDesignerNode)
  .addNode("builder_agent", builderAgentNode)
  .addNode("project_agent", projectAgentNode)
  .addNode("node_agent", nodeAgentNode)
  .addNode("data_agent", dataAgentNode)
  .addNode("document_agent", documentAgentNode)
  .addNode("media_agent", mediaAgentNode)
  .addNode("framework_agent", frameworkAgentNode)
  
  // Entry point
  .addEdge(START, "supervisor")
  
  // CopilotKit handler routes back to supervisor (NEW)
  .addEdge("copilotkit_handler", "supervisor")
  
  // Backend tool execution - check for pending frontend calls
  .addConditionalEdges("toolNode", async (state) => {
    if (state.pendingFrontendActions && state.pendingFrontendActions.length > 0) {
      console.log("  [toolNode] Pending frontend actions - routing to handler");
      return "copilotkit_handler";
    }
    return "supervisor";
  }, ["copilotkit_handler", "supervisor"])
  
  // Sub-agents route back to supervisor
  .addEdge("strategist", "supervisor")
  .addEdge("researcher", "supervisor")
  .addEdge("architect", "supervisor")
  .addEdge("writer", "supervisor")
  .addEdge("visual_designer", "supervisor")
  .addEdge("builder_agent", "supervisor")
  .addEdge("project_agent", "supervisor")
  .addEdge("node_agent", "supervisor")
  .addEdge("data_agent", "supervisor")
  .addEdge("document_agent", "supervisor")
  .addEdge("media_agent", "supervisor")
  .addEdge("framework_agent", "supervisor");

const graph = workflow.compile({
  checkpointer: postgresCheckpointer,
});
```

---

## Phase 6: Backend - Update All Sub-Agents

### 6.1 Sub-Agent Configuration Template

Apply this pattern to ALL sub-agents that may call CopilotKit tools:

```typescript
// =============================================================================
// SUB-AGENT TEMPLATE - Apply to all agents
// =============================================================================

import {
  copilotkitCustomizeConfig,
} from "@copilotkit/sdk-js/langgraph";

async function [agentName]Node(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log(`\n[${agentName}] Starting...`);
  
  // ... existing setup code ...
  
  // Configure CopilotKit for this agent's tool calls
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,   // Emit tool calls to frontend
    emitMessages: true,    // Emit message streaming
  });
  
  // Invoke LLM with custom config
  const response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    customConfig  // Use custom config instead of raw config
  );
  
  // ... rest of function ...
}
```

### 6.2 Sub-Agent Checklist

| Agent | File | Calls Frontend Tools? | Priority |
|-------|------|----------------------|----------|
| strategist | `agents/strategist.ts` | Review | Medium |
| researcher | `agents/researcher.ts` | No (web_search is backend) | Skip |
| architect | `agents/architect.ts` | Yes (`requestPlanApproval`) | **Critical** |
| writer | `agents/writer.ts` | Review | Medium |
| visual_designer | `agents/visual-designer.ts` | Review | Medium |
| builder_agent | `agents/builder-agent.ts` | Yes (creates artifacts) | **Critical** |
| project_agent | `agents/project-agent.ts` | Yes (`updateProject*`) | **Critical** |
| node_agent | `agents/node-agent.ts` | Yes (`createNode`, `updateNodeFields`) | **Critical** |
| data_agent | `agents/data-agent.ts` | Review | Medium |
| document_agent | `agents/document-agent.ts` | Review | Medium |
| media_agent | `agents/media-agent.ts` | Review | Medium |
| framework_agent | `agents/framework-agent.ts` | Review | Medium |

### 6.3 Critical Agent: Architect

```typescript
// agents/architect.ts - CRITICAL UPDATE

import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";

async function architectNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[architect] Planning course structure...");
  
  const model = new ChatOpenAI({ model: "gpt-4o" });
  
  // Architect calls requestPlanApproval - a frontend action
  const modelWithTools = model.bindTools([
    // ... architect tools including requestPlanApproval
  ]);
  
  // CRITICAL: Use CopilotKit config for proper tool emission
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
    emitMessages: true,
  });
  
  const response = await modelWithTools.invoke(
    [systemMessage, ...messages],
    customConfig  // This ensures requestPlanApproval is properly emitted
  );
  
  return {
    messages: [response],
    // ... other state updates
  };
}
```

### 6.4 Critical Agent: Node Agent

```typescript
// agents/node-agent.ts - CRITICAL UPDATE

import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";

async function nodeAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[node_agent] Managing course nodes...");
  
  const model = new ChatOpenAI({ model: "gpt-4o" });
  
  // Node agent calls createNode, updateNodeFields - frontend actions
  const modelWithTools = model.bindTools([
    // ... node tools
  ]);
  
  // CRITICAL: Use CopilotKit config
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,
    emitMessages: true,
  });
  
  const response = await modelWithTools.invoke(
    [systemMessage, ...messages],
    customConfig
  );
  
  return {
    messages: [response],
  };
}
```

---

## Phase 7: Frontend - CopilotKit Provider Updates

### 7.1 Update Provider with Thread Persistence

**Location:** `frontend/src/App.tsx` or provider file

```typescript
// =============================================================================
// COPILOTKIT PROVIDER - v1.50 with Thread Persistence
// =============================================================================

import { useState, useEffect } from "react";
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

// Thread ID persistence key
const THREAD_ID_KEY = "copilotkit_thread_id";

function App() {
  // Persist thread ID for conversation continuity
  const [threadId, setThreadId] = useState<string | undefined>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(THREAD_ID_KEY) || undefined;
    }
    return undefined;
  });
  
  // Save thread ID changes
  const handleThreadIdChange = (newThreadId: string) => {
    setThreadId(newThreadId);
    localStorage.setItem(THREAD_ID_KEY, newThreadId);
  };
  
  // Optional: Clear thread on explicit reset
  const handleNewConversation = () => {
    localStorage.removeItem(THREAD_ID_KEY);
    setThreadId(undefined);
  };
  
  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      agent="orchestrator"
      threadId={threadId}
      onThreadIdChange={handleThreadIdChange}
      // Optional: Show debug info in development
      showDevConsole={process.env.NODE_ENV === "development"}
    >
      {/* Your app content */}
      <YourAppContent onNewConversation={handleNewConversation} />
    </CopilotKit>
  );
}
```

### 7.2 Optional: useCoAgent for State Access

```typescript
// Access agent state from frontend
import { useCoAgent } from "@copilotkit/react-core";

interface OrchestratorAgentState {
  messages: any[];
  currentAgent: string | null;
  pendingFrontendActions: any[];
  // ... other state fields
}

function AgentStatePanel() {
  const { state, running } = useCoAgent<OrchestratorAgentState>({
    name: "orchestrator",
  });
  
  return (
    <div className="agent-state-panel">
      <div>Status: {running ? "Running" : "Idle"}</div>
      <div>Current Agent: {state?.currentAgent || "None"}</div>
      <div>Pending Actions: {state?.pendingFrontendActions?.length || 0}</div>
    </div>
  );
}
```

---

## Phase 8: Frontend - Update Action Hooks

### 8.1 Add `available: "remote"` to Agent-Triggered Actions

**Location:** Your action hooks file (e.g., `useCourseActions.ts`)

```typescript
// =============================================================================
// AGENT-TRIGGERED ACTIONS - Add available: "remote"
// =============================================================================

import { useCopilotAction } from "@copilotkit/react-core";

/**
 * Actions that are triggered by the agent, not the user.
 * The `available: "remote"` flag tells CopilotKit these actions
 * should only be called by the agent, not offered to the user.
 */

// Architect's plan approval
useCopilotAction({
  name: "requestPlanApproval",
  description: "Request user approval for the course plan",
  available: "remote", // Agent-triggered only
  parameters: [
    { name: "plan", type: "object", description: "The course plan to approve" },
    { name: "summary", type: "string", description: "Summary of the plan" },
  ],
  renderAndWaitForResponse: ({ args, respond, status }) => {
    if (status === "complete") return null;
    
    return (
      <PlanApprovalModal
        plan={args.plan}
        summary={args.summary}
        onApprove={() => respond?.({ approved: true, feedback: null })}
        onReject={(feedback) => respond?.({ approved: false, feedback })}
      />
    );
  },
});

// Generic action approval
useCopilotAction({
  name: "requestActionApproval",
  description: "Request user approval before performing a sensitive action",
  available: "remote",
  parameters: [
    { name: "action", type: "string", description: "The action to approve" },
    { name: "details", type: "object", description: "Action details" },
  ],
  renderAndWaitForResponse: ({ args, respond, status }) => {
    if (status === "complete") return null;
    
    return (
      <ConfirmationDialog
        action={args.action}
        details={args.details}
        onConfirm={() => respond?.({ confirmed: true })}
        onCancel={() => respond?.({ confirmed: false })}
      />
    );
  },
});

// Node creation
useCopilotAction({
  name: "createNode",
  description: "Create a new course node",
  available: "remote",
  parameters: [
    { name: "nodeType", type: "string", description: "Type of node" },
    { name: "content", type: "object", description: "Node content" },
  ],
  handler: async ({ nodeType, content }) => {
    const newNode = await createCourseNode(nodeType, content);
    return { success: true, nodeId: newNode.id };
  },
});

// Node updates
useCopilotAction({
  name: "updateNodeFields",
  description: "Update fields on an existing node",
  available: "remote",
  parameters: [
    { name: "nodeId", type: "string", description: "ID of the node to update" },
    { name: "fields", type: "object", description: "Fields to update" },
  ],
  handler: async ({ nodeId, fields }) => {
    await updateNodeFields(nodeId, fields);
    return { success: true };
  },
});

// Clarifying questions
useCopilotAction({
  name: "askClarifyingQuestions",
  description: "Ask the user clarifying questions",
  available: "remote",
  parameters: [
    { name: "questions", type: "array", description: "Questions to ask" },
  ],
  renderAndWaitForResponse: ({ args, respond, status }) => {
    if (status === "complete") return null;
    
    return (
      <QuestionsForm
        questions={args.questions}
        onSubmit={(answers) => respond?.({ answers })}
      />
    );
  },
});

// Options selection
useCopilotAction({
  name: "offerOptions",
  description: "Offer options for the user to choose from",
  available: "remote",
  parameters: [
    { name: "prompt", type: "string", description: "Prompt for the user" },
    { name: "options", type: "array", description: "Options to choose from" },
  ],
  renderAndWaitForResponse: ({ args, respond, status }) => {
    if (status === "complete") return null;
    
    return (
      <OptionsSelector
        prompt={args.prompt}
        options={args.options}
        onSelect={(selected) => respond?.({ selected })}
      />
    );
  },
});

// Progress display
useCopilotAction({
  name: "showProgress",
  description: "Show progress indicator to user",
  available: "remote",
  parameters: [
    { name: "stage", type: "string", description: "Current stage" },
    { name: "percent", type: "number", description: "Completion percentage" },
    { name: "message", type: "string", description: "Progress message" },
  ],
  handler: async ({ stage, percent, message }) => {
    // Update progress UI (via context or state)
    updateProgressIndicator({ stage, percent, message });
    return { acknowledged: true };
  },
});
```

### 8.2 Complete List of Actions to Update

| Action Name | Type | Add `available: "remote"`? |
|-------------|------|---------------------------|
| requestPlanApproval | HITL | ✅ Yes |
| requestActionApproval | HITL | ✅ Yes |
| askClarifyingQuestions | HITL | ✅ Yes |
| offerOptions | HITL | ✅ Yes |
| createNode | Handler | ✅ Yes |
| updateNodeFields | Handler | ✅ Yes |
| deleteNode | Handler | ✅ Yes |
| showProgress | Handler | ✅ Yes |
| displayPreview | Render | ✅ Yes |
| updateProject* | Handler | ✅ Yes |
| (any other agent-triggered actions) | - | ✅ Yes |

---

## Phase 9: Frontend - Debug Components

### 9.1 Tool Call Debugger (Development Only)

**Location:** Create `frontend/src/components/ToolCallDebugger.tsx`

```typescript
// =============================================================================
// TOOL CALL DEBUGGER - Development utility
// =============================================================================

import { useCopilotContext } from "@copilotkit/react-core";
import { useMemo } from "react";

interface ToolCall {
  id: string;
  name: string;
  args: any;
}

export function ToolCallDebugger() {
  const { messages } = useCopilotContext();
  
  // Find pending tool calls (calls without matching results)
  const { pendingToolCalls, recentToolResults } = useMemo(() => {
    const toolCalls: ToolCall[] = [];
    const toolResults: { id: string; name: string }[] = [];
    
    for (const msg of messages) {
      if (msg.role === "assistant" && msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          toolCalls.push({
            id: tc.id,
            name: tc.name,
            args: tc.args,
          });
        }
      }
      
      if (msg.role === "tool") {
        toolResults.push({
          id: msg.tool_call_id,
          name: msg.name,
        });
      }
    }
    
    // Pending = calls without matching results
    const resultIds = new Set(toolResults.map(r => r.id));
    const pending = toolCalls.filter(tc => !resultIds.has(tc.id));
    
    // Recent results (last 5)
    const recent = toolResults.slice(-5).reverse();
    
    return { pendingToolCalls: pending, recentToolResults: recent };
  }, [messages]);
  
  // Only show in development
  if (process.env.NODE_ENV !== "development") {
    return null;
  }
  
  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-sm">
      {/* Pending Tool Calls */}
      {pendingToolCalls.length > 0 && (
        <div className="bg-yellow-100 border border-yellow-400 rounded-lg p-3 mb-2 shadow-lg">
          <div className="font-semibold text-yellow-800 text-sm">
            ⚠️ Pending Tool Calls: {pendingToolCalls.length}
          </div>
          <div className="mt-1 space-y-1">
            {pendingToolCalls.map((tc) => (
              <div key={tc.id} className="text-xs text-yellow-700">
                • {tc.name} ({tc.id.slice(0, 8)}...)
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Recent Tool Results */}
      {recentToolResults.length > 0 && (
        <div className="bg-green-100 border border-green-400 rounded-lg p-3 shadow-lg">
          <div className="font-semibold text-green-800 text-sm">
            ✓ Recent Results
          </div>
          <div className="mt-1 space-y-1">
            {recentToolResults.map((tr) => (
              <div key={tr.id} className="text-xs text-green-700">
                • {tr.name}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* All clear */}
      {pendingToolCalls.length === 0 && recentToolResults.length === 0 && (
        <div className="bg-gray-100 border border-gray-300 rounded-lg p-2 shadow-lg">
          <div className="text-xs text-gray-500">No active tool calls</div>
        </div>
      )}
    </div>
  );
}
```

### 9.2 Agent Activity Indicator

```typescript
// =============================================================================
// AGENT ACTIVITY INDICATOR
// =============================================================================

import { useCoAgentStateRender } from "@copilotkit/react-core";

export function AgentActivityIndicator() {
  useCoAgentStateRender({
    name: "orchestrator",
    render: ({ state }) => {
      if (!state?.currentAgent || state.currentAgent === "orchestrator") {
        return null;
      }
      
      const agentLabels: Record<string, string> = {
        strategist: "🎯 Planning Strategy",
        researcher: "🔍 Researching",
        architect: "🏗️ Designing Structure",
        writer: "✍️ Writing Content",
        visual_designer: "🎨 Creating Visuals",
        builder_agent: "🔧 Building",
        project_agent: "📁 Managing Project",
        node_agent: "📝 Editing Nodes",
        data_agent: "📊 Processing Data",
        document_agent: "📄 Handling Documents",
        media_agent: "🎬 Processing Media",
        framework_agent: "⚙️ Configuring Framework",
      };
      
      const label = agentLabels[state.currentAgent] || `Working: ${state.currentAgent}`;
      
      return (
        <div className="fixed top-4 right-4 bg-blue-100 border border-blue-400 rounded-lg px-4 py-2 shadow-lg">
          <div className="flex items-center space-x-2">
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 rounded-full border-t-transparent" />
            <span className="text-blue-800 font-medium">{label}</span>
          </div>
        </div>
      );
    },
  });
  
  return null;
}
```

---

## Phase 10: Message Filtering

### 10.1 Filter Internal Agent Messages from UI

```typescript
// =============================================================================
// MESSAGE FILTERING - Hide internal agent messages
// =============================================================================

import { useCopilotContext } from "@copilotkit/react-core";
import { useMemo } from "react";

const INTERNAL_AGENT_NAMES = [
  "strategist",
  "researcher",
  "architect",
  "writer",
  "visual_designer",
  "builder_agent",
  "project_agent",
  "node_agent",
  "data_agent",
  "document_agent",
  "media_agent",
  "framework_agent",
];

export function useVisibleMessages() {
  const { messages } = useCopilotContext();
  
  return useMemo(() => {
    return messages.filter((msg) => {
      // Always show user messages
      if (msg.role === "user") return true;
      
      // Filter out internal agent messages
      const agentName = msg.name || msg.additional_kwargs?.name;
      if (agentName && INTERNAL_AGENT_NAMES.includes(agentName)) {
        return false;
      }
      
      // Show orchestrator/supervisor messages
      return true;
    });
  }, [messages]);
}

// Usage in chat component
function ChatMessages() {
  const visibleMessages = useVisibleMessages();
  
  return (
    <div className="chat-messages">
      {visibleMessages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
    </div>
  );
}
```

---

## Implementation Order

Execute in this order to minimize breaking changes:

| Step | Phase | Description | Risk |
|------|-------|-------------|------|
| 1 | 1 | Update package.json files (both) | Low |
| 2 | 2 | Add tool classification system | Low |
| 3 | 3 | Add copilotKitHandlerNode | Low |
| 4 | 4 | Update supervisor routing logic | **Medium** |
| 5 | 4 | Update safety check | Low |
| 6 | 5 | Update graph definition (add node, edges, `ends`) | **Medium** |
| 7 | 6 | Update architect agent (critical) | **Medium** |
| 8 | 6 | Update node_agent (critical) | **Medium** |
| 9 | 6 | Update remaining sub-agents | Low |
| 10 | 7 | Update frontend provider | Low |
| 11 | 8 | Add `available: "remote"` to actions | Low |
| 12 | 9 | Add debug components | Low |
| 13 | 10 | Add message filtering | Low |
| 14 | - | Full integration testing | - |

---

## Testing Checklist

### Backend Tests

- [ ] Tool classification correctly identifies backend vs frontend tools
- [ ] copilotKitHandlerNode emits tool calls and routes to supervisor
- [ ] Mixed tool calls (backend + frontend) handled correctly
- [ ] Safety check routes to handler instead of END
- [ ] All sub-agents use copilotkitCustomizeConfig
- [ ] Graph compiles without errors (check `ends` parameter)

### Frontend Tests

- [ ] Thread persistence works (refresh page, conversation continues)
- [ ] Plan approval flow completes without orphaned results
- [ ] Node creation flow completes without orphaned results
- [ ] Tool call debugger shows pending/completed calls correctly
- [ ] Agent activity indicator shows current agent
- [ ] Internal agent messages filtered from chat

### Integration Tests

| Scenario | Expected AG-UI Events |
|----------|----------------------|
| Create node via supervisor | `TOOL_CALL_START` → `TOOL_CALL_ARGS` → `TOOL_CALL_END` → `TOOL_RESULT` → `CONTINUE` → `RUN_FINISHED` |
| Architect plan approval | `TOOL_CALL_START` → ... → (user clicks approve) → `TOOL_RESULT` → `CONTINUE` → `RUN_FINISHED` |
| Mixed backend + frontend | Backend tools first, then frontend, then `RUN_FINISHED` |
| Multi-step workflow | Multiple tool call cycles, proper results at each step |

### Broken Pattern (Should Not Occur)

```
TOOL_CALL_START → TOOL_CALL_ARGS → TOOL_CALL_END → RUN_FINISHED → (tool result lost!)
```

---

## Rollback Plan

If issues occur after deployment:

1. **Quick Rollback:** Revert to routing CopilotKit tools to END (original behavior)
2. **Package Rollback:** Downgrade CopilotKit packages to previous version
3. **Partial Rollback:** Keep classification but skip handler node

---

## Success Metrics

After implementation, verify:

1. ✅ No orphaned tool calls in browser console
2. ✅ Multi-step workflows complete without user intervention
3. ✅ Architect approval flow works end-to-end
4. ✅ Node creation/update flows work end-to-end
5. ✅ Thread persistence maintains conversation state
6. ✅ No regression in existing functionality

---

## Appendix: Full File Diffs

### A.1 orchestrator-agent.ts Changes Summary

```diff
+ // Add after TABLE_TOOLS_FOR_DATA_AGENT (~line 136)
+ const BACKEND_TOOL_NAMES = new Set([...]);
+ const INTERNAL_ROUTING_TOOLS = new Set([...]);
+ function getCopilotKitActionNames(state) { ... }
+ function classifyToolCalls(toolCalls, state) { ... }

+ // Add after helper functions (~line 620)
+ async function copilotKitHandlerNode(state, config) { ... }

  // Update safety check (~line 646-678)
- return new Command({ goto: END, ... });
+ return new Command({ goto: "copilotkit_handler", ... });

  // Update routing after LLM response (~line 991-999)
- // Route to END for CopilotKit tools
- return new Command({ goto: END, ... });
+ // Route to handler for CopilotKit tools
+ return new Command({ goto: "copilotkit_handler", ... });

  // Update SUPERVISOR_ROUTING_DESTINATIONS
+ "copilotkit_handler",

  // Update graph definition
+ .addNode("copilotkit_handler", copilotKitHandlerNode)
+ .addEdge("copilotkit_handler", "supervisor")
  .addNode("supervisor", supervisorNode, {
+   ends: SUPERVISOR_ROUTING_DESTINATIONS,  // LangGraph 1.0.4 requirement
  })
```

---

## Document Version

- **Version:** 2.0
- **Last Updated:** December 2025
- **LangGraph Version:** 1.0.4
- **LangChain Core Version:** 1.1.4
- **CopilotKit Version:** 1.50.0
