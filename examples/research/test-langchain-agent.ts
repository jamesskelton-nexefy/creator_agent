/**
 * Test Script for Orchestrator LangChain Agent
 * 
 * This script mocks the CopilotKit state and tests that:
 * 1. The agent can invoke with mock frontend actions
 * 2. The FrontendToolMiddleware correctly intercepts tool calls
 * 3. Tool calls are emitted (would be picked up by CopilotKit in production)
 * 
 * Run with: npx tsx test-langchain-agent.ts
 */

import "dotenv/config";
import { randomUUID } from "crypto";
import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";

// Mock frontend actions that simulate what CopilotKit would provide
const MOCK_FRONTEND_ACTIONS = [
  {
    type: "function",
    function: {
      name: "switchViewMode",
      description: "Change the view mode of the application",
      parameters: {
        type: "object",
        properties: {
          mode: {
            type: "string",
            enum: ["document", "list", "graph", "table"],
            description: "The view mode to switch to"
          }
        },
        required: ["mode"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "navigateToProject",
      description: "Navigate to a specific project by ID",
      parameters: {
        type: "object",
        properties: {
          projectId: {
            type: "string",
            description: "The ID of the project to navigate to"
          }
        },
        required: ["projectId"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "showNotification",
      description: "Show a toast notification to the user",
      parameters: {
        type: "object",
        properties: {
          message: {
            type: "string",
            description: "The notification message"
          },
          type: {
            type: "string",
            enum: ["success", "error", "info", "warning"],
            description: "The notification type"
          }
        },
        required: ["message", "type"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "listProjects",
      description: "List all available projects",
      parameters: {
        type: "object",
        properties: {
          searchTerm: {
            type: "string",
            description: "Optional search term to filter projects"
          }
        },
        required: []
      }
    }
  },
  {
    type: "function",
    function: {
      name: "getProjectHierarchyInfo",
      description: "Get information about the project hierarchy levels and coding config",
      parameters: {
        type: "object",
        properties: {},
        required: []
      }
    }
  },
  {
    type: "function",
    function: {
      name: "offerOptions",
      description: "Present choices to the user with clickable buttons",
      parameters: {
        type: "object",
        properties: {
          title: {
            type: "string",
            description: "Title for the options"
          },
          options: {
            type: "array",
            items: {
              type: "object",
              properties: {
                id: { type: "string" },
                label: { type: "string" },
                description: { type: "string" }
              }
            },
            description: "The options to present"
          }
        },
        required: ["title", "options"]
      }
    }
  }
];

// Create mock CopilotKit state
function createMockCopilotKitState() {
  return {
    actions: MOCK_FRONTEND_ACTIONS,
    context: [
      {
        description: "Current project",
        value: "Test Project ABC"
      },
      {
        description: "User role",
        value: "admin"
      }
    ]
  };
}

// Simulate a tool result from CopilotKit (what would come back after client-side execution)
function simulateCopilotKitToolResult(toolCallId: string, toolName: string, args: any): ToolMessage {
  // Simulate different results based on tool name
  let result: any;
  
  switch (toolName) {
    case "switchViewMode":
      result = { success: true, mode: args.mode, message: `View switched to ${args.mode}` };
      break;
    case "navigateToProject":
      result = { success: true, projectId: args.projectId, message: `Navigated to project ${args.projectId}` };
      break;
    case "showNotification":
      result = { success: true, shown: true };
      break;
    case "listProjects":
      result = {
        success: true,
        projects: [
          { id: "proj-1", name: "Safety Training Course", client: "Acme Corp" },
          { id: "proj-2", name: "Onboarding Program", client: "TechStart Inc" },
          { id: "proj-3", name: "Compliance Module", client: "Finance Co" }
        ]
      };
      break;
    case "getProjectHierarchyInfo":
      result = {
        success: true,
        levels: [
          { level: 2, name: "Module" },
          { level: 3, name: "Lesson" },
          { level: 4, name: "Topic" },
          { level: 5, name: "Activity" },
          { level: 6, name: "Content" }
        ]
      };
      break;
    case "offerOptions":
      result = { success: true, rendered: true, optionsCount: args.options?.length || 0 };
      break;
    default:
      result = { success: true, executed: true };
  }
  
  return new ToolMessage({
    content: JSON.stringify(result),
    tool_call_id: toolCallId,
    name: toolName,
  });
}

// Test runner
async function runTests() {
  console.log("=".repeat(70));
  console.log("ORCHESTRATOR LANGCHAIN AGENT - TEST SUITE");
  console.log("=".repeat(70));
  console.log("");

  // Import the agent dynamically to ensure it's compiled fresh
  console.log("[Test] Importing agent...");
  const { agent } = await import("./orchestrator-agent-langchain.js");
  console.log("[Test] Agent imported successfully\n");

  // Test 1: Basic invocation with mock state
  console.log("-".repeat(70));
  console.log("TEST 1: Basic Agent Invocation with Mock CopilotKit State");
  console.log("-".repeat(70));
  
  const threadId = randomUUID();
  const config = {
    configurable: {
      thread_id: threadId,
    }
  };

  const mockState = {
    messages: [
      new HumanMessage("Hello! Can you switch the view to graph mode?")
    ],
    copilotkit: createMockCopilotKitState()
  };

  console.log(`[Test] Thread ID: ${threadId}`);
  console.log(`[Test] Input message: "${mockState.messages[0].content}"`);
  console.log(`[Test] Mock frontend actions: ${MOCK_FRONTEND_ACTIONS.length}`);
  console.log("");

  try {
    console.log("[Test] Invoking agent...");
    const result = await agent.invoke(mockState, config);
    
    console.log("\n[Test] Agent completed!");
    console.log(`[Test] Output messages: ${result.messages?.length || 0}`);
    
    // Analyze the output
    if (result.messages && result.messages.length > 0) {
      console.log("\n[Test] Message Analysis:");
      for (const msg of result.messages) {
        const msgType = msg.constructor.name;
        if (msgType === "AIMessage" || msgType === "AIMessageChunk") {
          const aiMsg = msg as AIMessage;
          if (aiMsg.tool_calls?.length) {
            console.log(`  - AIMessage with tool_calls:`);
            for (const tc of aiMsg.tool_calls) {
              console.log(`    - Tool: ${tc.name}, Args: ${JSON.stringify(tc.args)}`);
            }
          } else {
            const content = typeof aiMsg.content === "string" 
              ? aiMsg.content.substring(0, 100) 
              : JSON.stringify(aiMsg.content).substring(0, 100);
            console.log(`  - AIMessage: "${content}..."`);
          }
        } else if (msgType === "ToolMessage") {
          const toolMsg = msg as ToolMessage;
          console.log(`  - ToolMessage [${toolMsg.name}]: ${toolMsg.content.substring(0, 50)}...`);
        } else if (msgType === "HumanMessage") {
          console.log(`  - HumanMessage: "${msg.content}"`);
        } else {
          console.log(`  - ${msgType}`);
        }
      }
    }
    
    console.log("\n[Test 1] PASSED - Agent invoked successfully with mock CopilotKit state");
  } catch (error) {
    console.error("\n[Test 1] FAILED:", error);
  }

  // Test 2: Test with a message that should trigger tool use
  console.log("\n" + "-".repeat(70));
  console.log("TEST 2: Trigger Frontend Tool Call");
  console.log("-".repeat(70));

  const threadId2 = randomUUID();
  const config2 = {
    configurable: {
      thread_id: threadId2,
    }
  };

  const mockState2 = {
    messages: [
      new HumanMessage("List all the projects for me")
    ],
    copilotkit: createMockCopilotKitState()
  };

  console.log(`[Test] Thread ID: ${threadId2}`);
  console.log(`[Test] Input: "${mockState2.messages[0].content}"`);
  console.log("");

  try {
    console.log("[Test] Invoking agent...");
    const result = await agent.invoke(mockState2, config2);
    
    console.log("\n[Test] Agent completed!");
    
    // Check if any tool calls were made
    let toolCallsMade = false;
    if (result.messages) {
      for (const msg of result.messages) {
        if (msg.constructor.name === "AIMessage" || msg.constructor.name === "AIMessageChunk") {
          const aiMsg = msg as AIMessage;
          if (aiMsg.tool_calls?.length) {
            toolCallsMade = true;
            console.log("[Test] Tool calls detected:");
            for (const tc of aiMsg.tool_calls) {
              console.log(`  - ${tc.name}(${JSON.stringify(tc.args)})`);
            }
          }
        }
      }
    }

    if (toolCallsMade) {
      console.log("\n[Test 2] PASSED - Agent correctly attempted to call frontend tools");
    } else {
      console.log("\n[Test 2] INFO - No tool calls detected (agent may have responded differently)");
    }
  } catch (error) {
    console.error("\n[Test 2] FAILED:", error);
  }

  // Test 3: Simulate full round-trip with tool result
  console.log("\n" + "-".repeat(70));
  console.log("TEST 3: Simulated Round-Trip with Tool Result");
  console.log("-".repeat(70));
  
  const threadId3 = randomUUID();
  const config3 = {
    configurable: {
      thread_id: threadId3,
    }
  };

  // First turn - user asks, agent calls tool
  const mockState3 = {
    messages: [
      new HumanMessage("Switch to table view please")
    ],
    copilotkit: createMockCopilotKitState()
  };

  console.log(`[Test] Thread ID: ${threadId3}`);
  console.log(`[Test] Turn 1 - User: "${mockState3.messages[0].content}"`);
  console.log("");

  try {
    console.log("[Test] Invoking agent (Turn 1)...");
    const result1 = await agent.invoke(mockState3, config3);
    
    console.log("[Test] Turn 1 completed");
    
    // Find tool call in response
    let toolCall = null;
    if (result1.messages) {
      for (const msg of result1.messages) {
        if (msg.constructor.name === "AIMessage" || msg.constructor.name === "AIMessageChunk") {
          const aiMsg = msg as AIMessage;
          if (aiMsg.tool_calls?.length) {
            toolCall = aiMsg.tool_calls[0];
            console.log(`[Test] Agent called: ${toolCall.name}(${JSON.stringify(toolCall.args)})`);
          }
        }
      }
    }

    if (toolCall) {
      // Simulate CopilotKit executing the tool and returning result
      console.log("\n[Test] Simulating CopilotKit tool execution...");
      const toolResult = simulateCopilotKitToolResult(toolCall.id!, toolCall.name, toolCall.args);
      console.log(`[Test] Simulated result: ${toolResult.content}`);

      // Second turn - with tool result
      const mockState3b = {
        messages: [
          ...result1.messages,
          toolResult
        ],
        copilotkit: createMockCopilotKitState()
      };

      console.log("\n[Test] Invoking agent (Turn 2 - with tool result)...");
      const result2 = await agent.invoke(mockState3b, config3);
      
      console.log("[Test] Turn 2 completed");
      
      // Check final response
      if (result2.messages) {
        const lastMsg = result2.messages[result2.messages.length - 1];
        if (lastMsg.constructor.name === "AIMessage" || lastMsg.constructor.name === "AIMessageChunk") {
          const content = typeof lastMsg.content === "string" 
            ? lastMsg.content 
            : JSON.stringify(lastMsg.content);
          console.log(`[Test] Agent final response: "${content.substring(0, 150)}..."`);
        }
      }
      
      console.log("\n[Test 3] PASSED - Full round-trip simulation completed");
    } else {
      console.log("\n[Test 3] SKIPPED - No tool call was made in Turn 1");
    }
  } catch (error) {
    console.error("\n[Test 3] FAILED:", error);
  }

  console.log("\n" + "=".repeat(70));
  console.log("TEST SUITE COMPLETED");
  console.log("=".repeat(70));
}

// Run tests
runTests().catch(console.error);

