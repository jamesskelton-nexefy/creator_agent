# CopilotKit v1.50 + LangGraph Multi-Agent Integration Guide

## Nexefy Course Creator: Definitive Upgrade & Architecture Guide

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Critical Issues Identified](#3-critical-issues-identified)
4. [CopilotKit v1.50 Key Changes](#4-copilotkit-v150-key-changes)
5. [The Orphaned Tool Call Problem](#5-the-orphaned-tool-call-problem)
6. [Recommended Architecture Changes](#6-recommended-architecture-changes)
7. [Implementation Guide](#7-implementation-guide)
8. [Frontend Integration](#8-frontend-integration)
9. [Testing & Debugging](#9-testing--debugging)
10. [Migration Checklist](#10-migration-checklist)

---

## 1. Executive Summary

### What This Guide Covers

Your multi-agent course creation system uses a supervisor pattern with 12 sub-agents (strategist, researcher, architect, writer, visual_designer, builder_agent, and 6 tool-specialized agents). This guide addresses:

1. **Upgrading to CopilotKit v1.50+** - Thread persistence, useAgent v2 API
2. **Fixing orphaned tool calls** - The #1 issue with LangGraph + CopilotKit
3. **Optimizing frontend action routing** - Ensuring tool results don't get lost
4. **Best practices for your specific architecture**

### Key Findings

| Aspect | Current State | Recommended |
|--------|---------------|-------------|
| CopilotKit Version | Using `copilotkitCustomizeConfig` | Upgrade to v1.50+ with explicit `emitToolCalls` |
| Tool Routing | Routes to `END` for CopilotKit tools | Add dedicated CopilotKit handler node |
| Orphan Prevention | Safety check in supervisor | Implement full tool call classification |
| Thread Persistence | Manual via PostgresSaver | Use v1.50 built-in thread management |

---

## 2. Current Architecture Analysis

### Your Graph Structure

```
START → supervisor → [routing decision]
                          ↓
        ┌─────────────────┼───────────────────────────────────┐
        ↓                 ↓                 ↓                 ↓
   strategist        researcher        architect          ... (9 more)
        ↓                 ↓                 ↓                 ↓
        └─────────────────┼───────────────────────────────────┘
                          ↓
                     supervisor ← (loop back)
                          ↓
                         END (for CopilotKit tools or final response)
```

### Current Supervisor Routing Logic (orchestrator-agent.ts)

Your supervisor correctly identifies when the LLM emits tool calls but **routes all CopilotKit tool calls to END**:

```typescript
// Lines 991-999 in orchestrator-agent.ts
// CopilotKit frontend tool call (not supervisor_response) - route to END for CopilotKit to execute
console.log("  -> Routing to END (CopilotKit tool execution)");
return new Command({
  goto: END,
  update: {
    messages: updatedMessages,
    currentAgent: "orchestrator" as AgentType,
  },
});
```

**The Problem**: When you route to `END`, the LangGraph run finishes. CopilotKit executes the frontend action and returns a `ToolMessage`, but **there's no node listening** for that result. The result only appears when the next human message arrives.

### Your Safety Check (Lines 646-678)

You've implemented a safety check for unresolved tool calls:

```typescript
if (unresolvedCalls.length > 0) {
  console.log(`  [SAFETY] Last message has ${unresolvedCalls.length} unresolved tool_calls...`);
  console.log("  [SAFETY] Waiting for tool_result from CopilotKit - skipping LLM invocation");
  return new Command({
    goto: END,
    update: { currentAgent: "orchestrator" as AgentType },
  });
}
```

**This is good defensive code** but it's treating the symptom, not the cause. The graph shouldn't have ended with unresolved tool calls in the first place.

---

## 3. Critical Issues Identified

### Issue 1: Premature Run Termination

**Location**: `orchestrator-agent.ts` lines 991-999

**Symptom**: When the supervisor calls a CopilotKit frontend action (e.g., `createNode`, `offerOptions`), the graph routes to `END`. The run finishes, CopilotKit executes the action, but the result isn't processed until the next user message.

**Impact**:
- Tool results arrive "orphaned" in the next invocation
- Agents can't respond to tool results immediately
- Multi-step workflows break (e.g., architect calling `requestPlanApproval`)

### Issue 2: No Tool Call Classification

**Current Behavior**: All non-`supervisor_response` tool calls are treated the same way - routed to END.

**Missing Logic**: Separate classification of:
1. **Backend tools** (e.g., `web_search` in researcher) → Execute via ToolNode
2. **CopilotKit frontend tools** → Emit to frontend, keep run open

### Issue 3: Researcher's Backend Tools

**Location**: `orchestrator-agent.ts` lines 1044-1081

The researcher subgraph correctly handles backend tools internally:

```typescript
function createSubAgentGraphWithTools(
  nodeFunction: (state, config) => Promise<Partial<OrchestratorState>>,
  tools: StructuredToolInterface[]
) {
  const toolNode = new ToolNode(tools);
  // Routes to tools if backend tool calls present
}
```

**This is correct** - backend tools like `web_search` are executed within the subgraph before returning to supervisor. But CopilotKit tools called by sub-agents still have the same orphaning problem.

### Issue 4: Missing `emitToolCalls` Granularity

**Location**: `orchestrator-agent.ts` line 878

```typescript
const customConfig = copilotkitCustomizeConfig(config, {
  emitToolCalls: true,
});
```

You're emitting **all** tool calls to CopilotKit. This is fine, but without proper routing, they still get orphaned.

---

## 4. CopilotKit v1.50 Key Changes

### 4.1 New `useAgent` Hook (v2 API)

The v1.50 release introduces a new hook that better manages agent state and thread continuity:

```typescript
import { useAgent } from "@copilotkit/react-core/v2";

const { agent, sendMessage } = useAgent({ 
  agentId: "orchestrator" 
});

// Access state directly
const { state, messages, status } = agent;

// Messages and tool results are kept in sync
// Thread continuity is handled automatically
```

### 4.2 Thread Persistence

v1.50 has built-in thread management that works with your existing PostgresSaver:

```typescript
<CopilotKit 
  key={`${COPILOTKIT_MODE}-${activeThreadId || 'no-thread'}`}
  threadId={activeThreadId || undefined}
  agent="orchestrator"
>
```

Your current implementation in `App.tsx` is already correct! Thread persistence ensures tool calls and results stay attached to the same run.

### 4.3 Tool Call Lifecycle Improvements

v1.50 improves how tool calls flow:

1. **TOOL_CALL_START** → Agent emits tool call
2. **TOOL_CALL_ARGS** → Arguments stream
3. **TOOL_CALL_END** → Tool call complete
4. **Frontend executes** → Handler runs
5. **Tool result returned** → Result flows back to agent
6. **RUN_FINISHED** → Only after all results processed

**Critical**: For this to work, **your graph must not route to END when expecting tool results**.

### 4.4 `available: "remote"` for Agent-Triggered Actions

For actions that should only be called by the agent (not available to users directly):

```typescript
useCopilotAction({
  name: "requestPlanApproval",
  available: "remote",  // Only agent can trigger
  // ...
});
```

---

## 5. The Orphaned Tool Call Problem

### Root Cause Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ CURRENT FLOW (BROKEN)                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User: "Create a module for safety training"                    │
│                   ↓                                             │
│  Supervisor: Calls createNode (CopilotKit tool)                 │
│                   ↓                                             │
│  Router: goto: END  ← ❌ Run finishes here!                     │
│                   ↓                                             │
│  CopilotKit: Executes createNode on frontend                    │
│                   ↓                                             │
│  ToolMessage: { nodeId: "xyz123", success: true }               │
│                   ↓                                             │
│  NOWHERE TO GO!  ← ❌ Result is orphaned                        │
│                                                                 │
│  User: (must send another message)                              │
│                   ↓                                             │
│  Supervisor: Finally sees the ToolMessage                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Correct Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ CORRECT FLOW                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User: "Create a module for safety training"                    │
│                   ↓                                             │
│  Supervisor: Calls createNode (CopilotKit tool)                 │
│                   ↓                                             │
│  Router: goto: "copilotkit_handler"  ← ✅ Keep run open         │
│                   ↓                                             │
│  CopilotKitHandler: emits tool call, waits                      │
│                   ↓                                             │
│  CopilotKit: Executes createNode on frontend                    │
│                   ↓                                             │
│  ToolMessage: { nodeId: "xyz123", success: true }               │
│                   ↓                                             │
│  CopilotKitHandler: receives result, routes back                │
│                   ↓                                             │
│  Supervisor: Processes result, responds to user                 │
│                   ↓                                             │
│  END (only now!)  ← ✅ Run completes properly                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Recommended Architecture Changes

### 6.1 New Graph Structure

```
START → supervisor → route_tools (conditional)
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   sub_agents     copilotkit_handler       END
   (strategist,        ↓                   (final
    researcher,   [emits to frontend,      response
    architect,     waits for result]        only)
    etc.)              ↓
        ↓              ↓
        └──────────────┘
              ↓
          supervisor (loop)
```

### 6.2 Tool Classification Function

Add this to your orchestrator-agent.ts:

```typescript
// ============================================================================
// TOOL CLASSIFICATION
// ============================================================================

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
 * These are handled by ToolNode within subgraphs.
 */
const BACKEND_TOOL_NAMES = new Set([
  // Researcher tools
  "web_search",
  "listDocuments", 
  "searchDocuments",
  "searchDocumentsByText",
  "getDocumentLines",
  "getDocumentByName",
]);

/**
 * Classifies tool calls into backend (LangGraph) and frontend (CopilotKit).
 */
function classifyToolCalls(
  toolCalls: Array<{ name: string; id?: string; args: any }>,
  state: OrchestratorState
): {
  backendToolCalls: typeof toolCalls;
  frontendToolCalls: typeof toolCalls;
} {
  const copilotKitActions = getCopilotKitActionNames(state);
  
  const backendToolCalls: typeof toolCalls = [];
  const frontendToolCalls: typeof toolCalls = [];
  
  for (const toolCall of toolCalls) {
    // supervisor_response is our internal routing tool
    if (toolCall.name === "supervisor_response") {
      continue;
    }
    
    // Check if it's a known backend tool
    if (BACKEND_TOOL_NAMES.has(toolCall.name)) {
      backendToolCalls.push(toolCall);
    }
    // Check if it's a CopilotKit frontend action
    else if (copilotKitActions.has(toolCall.name)) {
      frontendToolCalls.push(toolCall);
    }
    // Unknown tools - assume CopilotKit (safer)
    else {
      console.warn(`  [supervisor] Unknown tool "${toolCall.name}" - treating as frontend`);
      frontendToolCalls.push(toolCall);
    }
  }
  
  return { backendToolCalls, frontendToolCalls };
}
```

### 6.3 CopilotKit Handler Node

Add a dedicated node that emits tool calls and keeps the run open:

```typescript
import { copilotkitCustomizeConfig, copilotkitEmitState } from "@copilotkit/sdk-js/langgraph";

// ============================================================================
// COPILOTKIT HANDLER NODE
// ============================================================================

/**
 * Handles CopilotKit frontend tool calls.
 * 
 * This node:
 * 1. Emits the tool calls to CopilotKit for frontend execution
 * 2. CopilotKit executes the actions on the frontend
 * 3. Tool results flow back through the message stream
 * 4. Routes back to supervisor to process results
 * 
 * CRITICAL: This node keeps the run open instead of ending prematurely.
 */
async function copilotKitHandlerNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Command> {
  console.log("\n[copilotkit-handler] ============ CopilotKit Handler ============");
  
  // Get pending frontend tool calls from the last AI message
  const messages = state.messages || [];
  const lastMessage = messages[messages.length - 1];
  const msgType = (lastMessage as any)._getType?.() || (lastMessage as any).constructor?.name || "";
  
  if (!(msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk")) {
    console.log("  [copilotkit-handler] No AI message to process");
    return new Command({ goto: "supervisor" });
  }
  
  const aiMessage = lastMessage as AIMessage;
  const toolCalls = aiMessage.tool_calls || [];
  
  if (toolCalls.length === 0) {
    console.log("  [copilotkit-handler] No tool calls to emit");
    return new Command({ goto: "supervisor" });
  }
  
  console.log(`  [copilotkit-handler] Emitting ${toolCalls.length} tool calls to CopilotKit`);
  console.log(`  [copilotkit-handler] Tools: ${toolCalls.map(tc => tc.name).join(", ")}`);
  
  // Configure to emit tool calls to CopilotKit
  const modifiedConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: toolCalls.map(tc => tc.name),
    emitMessages: true,
  });
  
  // Emit current state to CopilotKit
  await copilotkitEmitState(modifiedConfig, state);
  
  // Route back to supervisor to process tool results
  // The supervisor will either:
  // 1. Find tool results and process them
  // 2. Wait for more results (via the safety check)
  return new Command({
    goto: "supervisor",
    update: {
      currentAgent: "orchestrator" as AgentType,
    },
  });
}
```

### 6.4 Updated Supervisor Routing

Modify your supervisor's routing logic:

```typescript
// In supervisorNode, after handling supervisor_response routing...

// Handle non-routing tool calls (frontend actions)
if (aiResponse.tool_calls && aiResponse.tool_calls.length > 0) {
  // Check if supervisor_response was NOT called (frontend tool call)
  const hasRoutingTool = aiResponse.tool_calls.some(tc => tc.name === "supervisor_response");
  
  if (!hasRoutingTool) {
    // Classify the tool calls
    const { backendToolCalls, frontendToolCalls } = classifyToolCalls(
      aiResponse.tool_calls,
      state
    );
    
    console.log(`  Tool classification:`);
    console.log(`    Backend tools: ${backendToolCalls.map(tc => tc.name).join(", ") || "none"}`);
    console.log(`    Frontend tools: ${frontendToolCalls.map(tc => tc.name).join(", ") || "none"}`);
    
    // Route based on tool types
    if (frontendToolCalls.length > 0) {
      console.log("  -> Routing to copilotkit_handler (frontend tools)");
      return new Command({
        goto: "copilotkit_handler",
        update: {
          messages: updatedMessages,
          currentAgent: "orchestrator" as AgentType,
        },
      });
    }
    
    // If only backend tools (shouldn't happen from supervisor, but handle it)
    if (backendToolCalls.length > 0) {
      console.warn("  [supervisor] Backend tools called from supervisor - this is unusual");
      // Could route to a backend tool node if needed
    }
  }
}
```

### 6.5 Updated Graph Definition

```typescript
// Update SUPERVISOR_ROUTING_DESTINATIONS
const SUPERVISOR_ROUTING_DESTINATIONS = [
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
  // CopilotKit handler
  "copilotkit_handler",
  // End (final response only)
  END,
];

const workflow = new StateGraph(OrchestratorStateAnnotation)
  // Supervisor node - uses Command for routing
  .addNode("supervisor", supervisorNode, { 
    ends: SUPERVISOR_ROUTING_DESTINATIONS 
  })
  
  // NEW: CopilotKit handler for frontend tool calls
  .addNode("copilotkit_handler", copilotKitHandlerNode)
  
  // ... existing sub-agent nodes ...
  
  // Entry point -> supervisor
  .addEdge(START, "supervisor")
  
  // CopilotKit handler routes back to supervisor
  .addEdge("copilotkit_handler", "supervisor")
  
  // ... existing sub-agent edges ...
```

---

## 7. Implementation Guide

### Step 1: Update Dependencies

```bash
# In your frontend project
npm install @copilotkit/react-core@latest @copilotkit/react-ui@latest

# In your LangGraph project  
npm install @copilotkit/sdk-js@latest @langchain/langgraph@latest
```

### Step 2: Add Tool Classification

Add the `classifyToolCalls` function from Section 6.2 to your `orchestrator-agent.ts`.

### Step 3: Add CopilotKit Handler Node

Add the `copilotKitHandlerNode` function from Section 6.3.

### Step 4: Update Graph Definition

1. Add `"copilotkit_handler"` to `SUPERVISOR_ROUTING_DESTINATIONS`
2. Add `.addNode("copilotkit_handler", copilotKitHandlerNode)`
3. Add `.addEdge("copilotkit_handler", "supervisor")`

### Step 5: Update Supervisor Routing Logic

Replace the current "route to END for CopilotKit" logic (lines 991-999) with the classification-based routing from Section 6.4.

### Step 6: Update Sub-Agent Tool Emission

Each sub-agent that calls CopilotKit tools should also use `copilotkitCustomizeConfig`:

```typescript
// In each sub-agent node (e.g., architectNode, writerNode, etc.)

export async function architectNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  // ... existing setup ...
  
  // Configure CopilotKit emission for this agent's tools
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true,  // Emit all tool calls
    emitMessages: true,
  });
  
  const response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    customConfig  // Use the custom config
  );
  
  // ... rest of the function ...
}
```

### Step 7: Handle Sub-Agent CopilotKit Tools

For sub-agents that call CopilotKit tools (architect, node_agent, etc.), update their subgraphs:

```typescript
function createSubAgentGraphWithCopilotKit(
  nodeFunction: (state: OrchestratorState, config: RunnableConfig) => Promise<Partial<OrchestratorState>>
) {
  // Simple subgraph that returns to supervisor
  // The supervisor's copilotkit_handler will emit any pending tool calls
  return new StateGraph(OrchestratorStateAnnotation)
    .addNode("agent" as any, nodeFunction)
    .addEdge(START, "agent" as any)
    .addEdge("agent" as any, END)
    .compile();
}
```

---

## 8. Frontend Integration

### 8.1 Update App.tsx for v1.50

Your current setup is mostly correct. Key additions:

```typescript
import { CopilotKit } from '@copilotkit/react-core';

function CopilotKitWithThread({ children }: { children: ReactNode }) {
  const { activeThreadId } = useChatThreadContext();
  
  const copilotProps = COPILOTKIT_MODE === 'local' 
    ? { runtimeUrl: COPILOTKIT_RUNTIME_URL }
    : { publicLicenseKey: COPILOTKIT_PUBLIC_KEY };
  
  return (
    <CopilotKit 
      key={`${COPILOTKIT_MODE}-${activeThreadId || 'no-thread'}`}
      {...copilotProps}
      threadId={activeThreadId || undefined}
      agent="orchestrator"
      showDevConsole={true}
      // NEW v1.50: Better error handling
      onError={(errorEvent) => {
        console.error('[CopilotKit Error]', {
          type: errorEvent.type,
          source: errorEvent.context?.source,
          agent: errorEvent.context?.agent,
        });
      }}
    >
      {children}
    </CopilotKit>
  );
}
```

### 8.2 Frontend Actions Best Practices

For actions that the agent calls (not user-initiated):

```typescript
import { useCopilotAction } from "@copilotkit/react-core";

// Plan approval - agent-triggered
useCopilotAction({
  name: "requestPlanApproval",
  description: "Request user approval for a proposed plan",
  available: "remote",  // Only agent can trigger
  parameters: [
    { name: "plan", type: "string", required: true },
    { name: "title", type: "string", required: false },
  ],
  renderAndWaitForResponse: ({ args, respond, status }) => {
    if (status === "complete") return null;
    
    return (
      <PlanApprovalDialog
        plan={args.plan}
        title={args.title}
        onApprove={() => respond?.({ approved: true })}
        onReject={(reason) => respond?.({ approved: false, reason })}
      />
    );
  },
});

// Node creation - can be called by agent or user
useCopilotAction({
  name: "createNode",
  description: "Create a new content node",
  parameters: [
    { name: "templateId", type: "string", required: true },
    { name: "title", type: "string", required: true },
    { name: "parentNodeId", type: "string", required: false },
  ],
  handler: async ({ templateId, title, parentNodeId }) => {
    const result = await nodeService.createNode(templateId, title, parentNodeId);
    return { 
      success: true, 
      nodeId: result.id,
      message: `Created node "${title}"` 
    };
  },
});
```

### 8.3 Using the v2 useAgent Hook

For more control over agent state:

```typescript
import { useAgent } from "@copilotkit/react-core/v2";

function AgentStatus() {
  const { agent, sendMessage } = useAgent({ agentId: "orchestrator" });
  
  // Real-time status
  const isThinking = agent.status === "running";
  
  // Access all messages including tool calls/results
  const messages = agent.messages;
  
  // Find pending tool calls
  const pendingToolCalls = messages
    .filter(m => m.role === "assistant" && m.tool_calls?.length)
    .flatMap(m => m.tool_calls)
    .filter(tc => !messages.some(m => m.tool_call_id === tc.id));
  
  return (
    <div>
      {isThinking && <Spinner />}
      {pendingToolCalls.length > 0 && (
        <div>Executing: {pendingToolCalls.map(tc => tc.name).join(", ")}</div>
      )}
    </div>
  );
}
```

---

## 9. Testing & Debugging

### 9.1 AG-UI Event Monitoring

In browser dev tools, watch for these events:

```javascript
// Good flow
TOOL_CALL_START → TOOL_CALL_ARGS → TOOL_CALL_END → (frontend executes) → TOOL_RESULT → CONTINUE → RUN_FINISHED

// Broken flow (orphaned)
TOOL_CALL_START → TOOL_CALL_ARGS → TOOL_CALL_END → RUN_FINISHED → (tool result lost!)
```

### 9.2 Debug Logging

Add comprehensive logging in your supervisor:

```typescript
// After classifying tool calls
console.log("=== TOOL CLASSIFICATION ===");
console.log("Raw tool calls:", aiResponse.tool_calls?.map(tc => tc.name));
console.log("CopilotKit actions available:", [...getCopilotKitActionNames(state)]);
console.log("Backend tools:", backendToolCalls.map(tc => tc.name));
console.log("Frontend tools:", frontendToolCalls.map(tc => tc.name));
console.log("Routing decision:", frontendToolCalls.length > 0 ? "copilotkit_handler" : "END");
```

### 9.3 Tool Result Tracking

Add a utility to track tool call lifecycle:

```typescript
function logToolCallState(messages: BaseMessage[]) {
  const toolCalls: Map<string, { name: string; hasResult: boolean }> = new Map();
  
  for (const msg of messages) {
    const msgType = getMessageType(msg);
    
    if (msgType === "ai" || msgType === "AIMessage") {
      const aiMsg = msg as AIMessage;
      for (const tc of aiMsg.tool_calls || []) {
        if (tc.id) {
          toolCalls.set(tc.id, { name: tc.name, hasResult: false });
        }
      }
    }
    
    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      const existing = toolCalls.get(toolMsg.tool_call_id);
      if (existing) {
        existing.hasResult = true;
      }
    }
  }
  
  console.log("=== TOOL CALL STATE ===");
  for (const [id, state] of toolCalls) {
    console.log(`  ${state.name} (${id.slice(0, 8)}...): ${state.hasResult ? "✓ resolved" : "⏳ pending"}`);
  }
}
```

### 9.4 Common Issues & Solutions

| Symptom | Cause | Solution |
|---------|-------|----------|
| Tool result appears with next message | Run ended prematurely | Route to `copilotkit_handler` instead of `END` |
| "Orphaned tool result" warnings | ToolMessage without matching tool_use | Check `filterOrphanedToolResults` is running |
| Agent doesn't respond to tool result | Run finished before result arrived | Keep run open with handler node |
| Duplicate tool calls | `emitToolCalls: true` emits twice | Use specific tool names array instead |
| Safety check always triggers | Unresolved tool calls accumulating | Fix routing to process results |

---

## 10. Migration Checklist

### Pre-Migration

- [ ] Backup current codebase
- [ ] Document current behavior for regression testing
- [ ] Identify all places where CopilotKit tools are called

### Code Changes

- [ ] Add `classifyToolCalls` function
- [ ] Add `copilotKitHandlerNode` 
- [ ] Update `SUPERVISOR_ROUTING_DESTINATIONS`
- [ ] Update graph definition with new node and edge
- [ ] Replace END routing with handler routing for frontend tools
- [ ] Add `copilotkitCustomizeConfig` to sub-agents
- [ ] Update npm dependencies

### Testing

- [ ] Test each sub-agent's tool calls individually
- [ ] Test multi-step workflows (e.g., architect plan approval)
- [ ] Test tool call interruption/resumption
- [ ] Test thread persistence across page reloads
- [ ] Monitor AG-UI events in dev tools

### Frontend

- [ ] Update CopilotKit packages to v1.50+
- [ ] Add `available: "remote"` to agent-triggered actions
- [ ] Verify `renderAndWaitForResponse` actions work correctly
- [ ] Test `useAgent` v2 hook if using advanced features

### Post-Migration

- [ ] Remove temporary safety workarounds if no longer needed
- [ ] Update documentation
- [ ] Monitor production for orphaned tool call warnings

---

## Appendix A: Complete File Changes Summary

### orchestrator-agent.ts

1. **Add imports**:
   ```typescript
   import { copilotkitCustomizeConfig, copilotkitEmitState } from "@copilotkit/sdk-js/langgraph";
   ```

2. **Add tool classification** (new section ~line 135)

3. **Add copilotkit_handler node** (new section ~line 1015)

4. **Update SUPERVISOR_ROUTING_DESTINATIONS** (add "copilotkit_handler")

5. **Update routing logic** (replace lines 991-999 with classification-based routing)

6. **Update graph definition** (add node and edge)

### Sub-agent files (architect.ts, writer.ts, etc.)

1. **Add import**:
   ```typescript
   import { copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";
   ```

2. **Wrap config in model invocation**:
   ```typescript
   const customConfig = copilotkitCustomizeConfig(config, {
     emitToolCalls: true,
   });
   const response = await modelWithTools.invoke([...], customConfig);
   ```

### App.tsx

1. **Verify v1.50 compatibility** (mostly already correct)
2. **Consider adding useAgent v2 hook** for advanced state management

---

## Appendix B: Quick Reference

### Tool Call Flow Decision Tree

```
LLM emits tool_call(s)
          ↓
Is it supervisor_response?
    YES → Route to specified next_agent or END
    NO  ↓
          ↓
Classify tool calls
          ↓
Any frontend tools?
    YES → Route to copilotkit_handler
    NO  ↓
          ↓
Any backend tools?
    YES → Route to tool_node (shouldn't happen from supervisor)
    NO  → Route to END (no tool calls to process)
```

### Key CopilotKit SDK Functions

```typescript
// Configure which tool calls to emit
copilotkitCustomizeConfig(config, {
  emitToolCalls: true | string[],  // Emit all or specific tools
  emitMessages: boolean,            // Emit messages to frontend
});

// Emit current state to CopilotKit
await copilotkitEmitState(config, state);

// Get CopilotKit actions from state
state.copilotkit?.actions  // Array of available frontend actions
```

---

*Guide version: 1.0*
*Last updated: December 2024*
*Compatible with: CopilotKit v1.50+, LangGraph.js v0.2+*
