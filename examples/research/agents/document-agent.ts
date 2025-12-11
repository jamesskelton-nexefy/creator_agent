/**
 * Document Agent
 *
 * Specialized agent for document operations and RAG (Retrieval Augmented Generation).
 * Handles uploading documents, searching document content, and listing available documents.
 *
 * Tools (Frontend):
 * - uploadDocument - Trigger document upload dialog
 * - listDocuments - List available documents
 * - searchDocuments - Semantic search across documents
 * - searchDocumentsByText - Keyword/exact text search
 * - getDocumentByName - Get document by name
 * - getDocumentLines - Get specific lines from a document
 *
 * Output: Responds directly to user with document information
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import type { OrchestratorState } from "../state/agent-state";
import { generateTaskContext } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  stripThinkingBlocks,
  hasUsableResponse,
} from "../utils";

// Import tool categories for reference
import { TOOL_CATEGORIES } from "../orchestrator-agent-deep";

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const documentAgentModel = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
  maxTokens: 8000,
  temperature: 0.3, // Lower temperature for precise search results
});

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const DOCUMENT_AGENT_SYSTEM_PROMPT = `You are The Document Agent - a specialized agent for document search and RAG operations.

## Your Role

You help users find information in uploaded documents. You can:
- Search documents using semantic (meaning-based) search
- Search documents using keyword/exact text search
- List available documents
- Help users upload new documents
- Retrieve specific content from documents

## Your Tools

### Document Discovery
- **listDocuments(category?, limit?)** - List available documents
  - category: "course_content" or "framework_content"
  - Shows document names, types, and upload dates

### Semantic Search (Best for concepts and questions)
- **searchDocuments(query, category?, limit?)** - Search by meaning
  - Best for: longer queries, conceptual questions
  - Examples: "how to perform vehicle inspections", "safety procedures for heavy vehicles"
  - Returns relevant passages with context

### Keyword Search (Best for specific terms)
- **searchDocumentsByText(searchText, category?, limit?)** - Search by exact text
  - Best for: single words, acronyms, specific phrases
  - Examples: "GVM", "HVNL", "pre-trip", "chain of responsibility"
  - Returns matches with surrounding context

### Document Upload
- **uploadDocument(category, instructions?)** - Trigger upload dialog
  - category: "course_content" or "framework_content"
  - Supported formats: PDF, DOCX

### Content Retrieval
- **getDocumentByName(name)** - Get document metadata by name
- **getDocumentLines(documentId, startLine, endLine)** - Get specific lines

## Search Strategy

1. **Start with semantic search** for conceptual queries
   - Good: "what are the requirements for driver fatigue management"
   - searchDocuments({ query: "driver fatigue management requirements" })

2. **Use keyword search** for specific terms or if semantic returns nothing
   - Good: "GVM" or "HVNL" or "pre-trip inspection"
   - searchDocumentsByText({ searchText: "GVM" })

3. **Combine both** for comprehensive results
   - First semantic for context, then keyword for specific terms

## Communication Style

- Summarize search results clearly
- Quote relevant passages when helpful
- Indicate which document the information came from
- If no results found, suggest alternative search terms
- Offer to search with different approaches

## When to Hand Back

Include \`[DONE]\` in your response when:
- You've provided the requested document information
- The user's question has been answered from documents
- The user wants to do something outside document search

Example: "Based on the uploaded documents, the pre-trip inspection should include... [DONE]"`;

// ============================================================================
// DOCUMENT AGENT NODE FUNCTION
// ============================================================================

/**
 * The Document Agent node.
 * Handles document search and RAG operations.
 */
export async function documentAgentNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[document-agent] ============ Document Agent ============");

  // Get frontend tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  
  // Document tools
  const documentToolNames = TOOL_CATEGORIES.document;
  
  const documentAgentTools = frontendActions.filter((action: { name: string }) =>
    documentToolNames.includes(action.name)
  );

  console.log("  Available tools:", documentAgentTools.map((t: { name: string }) => t.name).join(", ") || "none");
  console.log("  Total tools:", documentAgentTools.length);

  // Build context-aware system message
  let systemContent = DOCUMENT_AGENT_SYSTEM_PROMPT;

  // Add task context for continuity across context trimming
  const taskContext = generateTaskContext(state);
  if (taskContext) {
    systemContent += `\n\n${taskContext}`;
  }

  // If we have research context, add it
  if (state.researchFindings) {
    systemContent += `\n\n## Previous Research Findings
The researcher has already gathered this information:
${JSON.stringify(state.researchFindings, null, 2).substring(0, 1500)}...

Use document search to find additional details or verify information.`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Bind tools and invoke
  const modelWithTools = documentAgentTools.length > 0
    ? documentAgentModel.bindTools(documentAgentTools)
    : documentAgentModel;

  // Filter messages for this agent's context
  const strippedMessages = stripThinkingBlocks(state.messages || []);
  const slicedMessages = strippedMessages.slice(-12);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[document-agent]");

  console.log("  Invoking document agent model...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Document agent response received");

  let aiResponse = response as AIMessage;
  if (aiResponse.tool_calls?.length) {
    console.log("  Tool calls:", aiResponse.tool_calls.map((tc) => tc.name).join(", "));
  }

  // RETRY LOGIC: If response is empty/thinking-only, retry with a nudge
  if (!hasUsableResponse(aiResponse)) {
    console.log("  [RETRY] Empty response detected - retrying with nudge message...");
    
    const nudgeMessage = new HumanMessage({
      content: `[SYSTEM NUDGE] The previous response was empty. Please respond now:
1. If the user wants to search documents, call searchDocuments or searchDocumentsByText
2. If listing documents, call listDocuments
3. Provide helpful results with relevant quotes

The user is waiting for document information.`,
    });

    console.log("  [RETRY] Re-invoking with nudge...");
    response = await modelWithTools.invoke(
      [systemMessage, ...recentMessages, nudgeMessage],
      config
    );
    
    aiResponse = response as AIMessage;
    
    if (hasUsableResponse(aiResponse)) {
      console.log("  [RETRY] Success - got usable response on retry");
    }
  }

  return {
    messages: [response],
    currentAgent: "document_agent",
    agentHistory: ["document_agent"],
    routingDecision: null,
  };
}

export default documentAgentNode;








