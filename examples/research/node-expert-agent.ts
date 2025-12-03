/**
 * Node Expert Agent
 *
 * A specialized LangGraph agent for creating content node structures.
 * Uses Claude 3.7 Sonnet with extended thinking to reason about
 * appropriate content for each hierarchy level.
 *
 * Key features:
 * - Inherits CopilotKit state to access frontend tools
 * - Uses extended thinking for deep reasoning about content structure
 * - Can call frontend tools (createNode, getAvailableTemplates, etc.)
 *
 * Based on CopilotKit frontend actions pattern:
 * https://docs.copilotkit.ai/langgraph/frontend-actions
 */

import "dotenv/config";
import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { START, StateGraph, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { Annotation } from "@langchain/langgraph";
import { CopilotKitStateAnnotation, copilotkitCustomizeConfig } from "@copilotkit/sdk-js/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import OpenAI from "openai";
import { z } from "zod";
import { createDocumentService, DocumentService } from "../../src/documents/index";

// ============================================================================
// MESSAGE FILTERING - Fix orphaned tool results and empty messages
// ============================================================================

/**
 * Checks if an AI message has non-empty text content.
 * Messages with only thinking blocks (no text) cause Anthropic API errors.
 */
function hasNonEmptyTextContent(msg: AIMessage): boolean {
  const content = msg.content;
  
  // String content - check if non-empty
  if (typeof content === 'string') {
    return content.trim().length > 0;
  }
  
  // Array content - look for non-empty text blocks
  if (Array.isArray(content)) {
    for (const block of content) {
      if (typeof block === 'string' && block.trim().length > 0) {
        return true;
      }
      if (typeof block === 'object' && block !== null) {
        // Check for text block with content
        if ('type' in block && block.type === 'text' && 'text' in block) {
          const text = (block as any).text;
          if (typeof text === 'string' && text.trim().length > 0) {
            return true;
          }
        }
      }
    }
  }
  
  // If message has tool_calls, it's valid even without text
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    return true;
  }
  
  return false;
}

/**
 * Filters out orphaned messages to prevent Anthropic API errors:
 * - tool_result without matching tool_use
 * - AI messages with ONLY unresolved tool_calls and no text content
 * 
 * IMPORTANT: When an AI message has SOME resolved and SOME unresolved tool_calls,
 * we create a NEW AI message with only the resolved ones to preserve the tool_use
 * for existing tool_results.
 */
function filterOrphanedToolResults(messages: BaseMessage[]): BaseMessage[] {
  const filtered: BaseMessage[] = [];
  
  // First pass: collect all tool_result IDs
  const toolResultIds = new Set<string>();
  for (const msg of messages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
    if (msgType === 'tool' || msgType === 'ToolMessage') {
      toolResultIds.add((msg as ToolMessage).tool_call_id);
    }
  }
  
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
    
    // Handle AI messages
    if (msgType === 'ai' || msgType === 'AIMessage' || msgType === 'AIMessageChunk') {
      const aiMsg = msg as AIMessage;
      
      // Check for tool_calls that need filtering
      if (aiMsg.tool_calls?.length) {
        const resolvedToolCalls = aiMsg.tool_calls.filter(tc => tc.id && toolResultIds.has(tc.id));
        const unresolvedToolCalls = aiMsg.tool_calls.filter(tc => tc.id && !toolResultIds.has(tc.id));
        
        if (unresolvedToolCalls.length > 0) {
          console.log(`  [FILTER] Found ${unresolvedToolCalls.length} unresolved tool_calls: ${unresolvedToolCalls.map(tc => tc.name).join(', ')}`);
        }
        
        // If we have resolved tool_calls, create a new AI message with only those
        if (resolvedToolCalls.length > 0) {
          if (unresolvedToolCalls.length > 0) {
            const newAiMsg = new AIMessage({
              content: aiMsg.content,
              tool_calls: resolvedToolCalls,
              id: aiMsg.id,
              name: aiMsg.name,
              additional_kwargs: { ...aiMsg.additional_kwargs },
              response_metadata: aiMsg.response_metadata,
            });
            filtered.push(newAiMsg);
            console.log(`  [FILTER] Kept AI message with ${resolvedToolCalls.length} resolved tool_calls`);
          } else {
            filtered.push(msg);
          }
          continue;
        }
      }
      
      // No resolved tool_calls - check for text content
      if (!hasNonEmptyTextContent(aiMsg)) {
        console.log(`  [FILTER] Removing AI message with no content and no resolved tool_calls`);
        continue;
      }
      filtered.push(msg);
      continue;
    }
    
    // Handle tool messages - check for orphaned tool_results
    if (msgType === 'tool' || msgType === 'ToolMessage') {
      const toolMsg = msg as ToolMessage;
      const toolCallId = toolMsg.tool_call_id;
      
      // Search ALL previous AI messages for matching tool_use
      let hasMatchingToolUse = false;
      for (let j = filtered.length - 1; j >= 0; j--) {
        const prevMsg = filtered[j];
        const prevType = (prevMsg as any)._getType?.() || (prevMsg as any).constructor?.name || '';
        
        if (prevType === 'ai' || prevType === 'AIMessage' || prevType === 'AIMessageChunk') {
          const aiMsg = prevMsg as AIMessage;
          if (aiMsg.tool_calls?.some(tc => tc.id === toolCallId)) {
            hasMatchingToolUse = true;
            break; // Found matching tool_use, stop searching
          }
          // Continue searching older AI messages (don't break here)
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

// ============================================================================
// STATE DEFINITION
// ============================================================================

/**
 * Tracks a node that has been created in this session.
 * Used to prevent duplicate creation.
 */
interface CreatedNode {
  parentNodeId: string | null;
  title: string;
  nodeId: string;
  templateName: string;
}

/**
 * Agent state that inherits from CopilotKitStateAnnotation.
 * This gives us access to:
 * - messages: Chat history
 * - copilotkit.actions: Frontend tools from useFrontendTool hooks
 * - copilotkit.context: Context from useCopilotReadable hooks
 * 
 * Extended with:
 * - createdNodes: Tracks nodes created in this session to prevent duplicates
 */
const NodeExpertStateAnnotation = Annotation.Root({
  ...CopilotKitStateAnnotation.spec,
  // Track created nodes to prevent duplicates
  createdNodes: Annotation<CreatedNode[]>({
    reducer: (existing, update) => {
      // Merge new nodes with existing, avoiding duplicates by nodeId
      const merged = [...(existing || [])];
      for (const node of (update || [])) {
        if (!merged.some(n => n.nodeId === node.nodeId)) {
          merged.push(node);
        }
      }
      return merged;
    },
    default: () => [],
  }),
});

export type NodeExpertState = typeof NodeExpertStateAnnotation.State;

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

/**
 * Claude 3.7 Sonnet with extended thinking enabled.
 * Extended thinking allows the model to "think" through complex
 * content structure decisions before responding.
 */
const model = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7,
});

// ============================================================================
// PERPLEXITY WEB SEARCH TOOL
// ============================================================================

/**
 * Perplexity client configured with OpenAI SDK compatibility.
 * Uses the Perplexity API endpoint with sonar-pro model for web search.
 * Reference: https://docs.perplexity.ai/getting-started/quickstart
 */
const perplexityClient = new OpenAI({
  apiKey: process.env.PERPLEXITY_API_KEY,
  baseURL: "https://api.perplexity.ai",
});

/**
 * Web search tool using Perplexity Sonar Pro API.
 * Provides real-time web search capabilities with citations.
 */
const webSearch = tool(
  async ({ query }: { query: string }) => {
    console.log(`  [web_search] Searching for: "${query}"`);
    
    try {
      const response = await perplexityClient.chat.completions.create({
        model: "sonar-pro",
        messages: [{ role: "user", content: query }],
      });

      // Extract content and citations from response
      const content = response.choices[0]?.message?.content || "No results found.";
      // Perplexity returns citations in a custom field (cast to access it)
      const citations = (response as any).citations || [];

      console.log(`  [web_search] Got response with ${citations.length} citations`);

      return JSON.stringify({
        answer: content,
        citations: citations,
        query: query,
      });
    } catch (error) {
      console.error(`  [web_search] Error:`, error);
      return JSON.stringify({
        error: `Web search failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        query: query,
      });
    }
  },
  {
    name: "web_search",
    description: "Search the web for current information using Perplexity AI. Use this to research topics, find recent events, look up facts, or gather information before creating content nodes.",
    schema: z.object({
      query: z.string().describe("The search query to look up on the web"),
    }),
  }
);

// ============================================================================
// DOCUMENT RAG TOOLS
// ============================================================================

/**
 * Document service for RAG operations.
 * Lazily initialized to avoid errors when env vars are not set.
 */
let documentService: DocumentService | null = null;

function getDocumentService(): DocumentService {
  if (!documentService) {
    documentService = createDocumentService();
  }
  return documentService;
}

/**
 * List available documents in the system.
 */
const listDocuments = tool(
  async ({ category, orgId, projectId, limit }: { 
    category?: string; 
    orgId?: string; 
    projectId?: string;
    limit?: number;
  }) => {
    console.log(`  [listDocuments] Listing documents - category: ${category || 'all'}, orgId: ${orgId || 'all'}`);
    
    try {
      const docs = await getDocumentService().listDocuments({
        category: category as "course_content" | "framework_content" | undefined,
        orgId,
        projectId,
        limit: limit || 20,
      });

      console.log(`  [listDocuments] Found ${docs.length} documents`);

      return JSON.stringify({
        success: true,
        count: docs.length,
        documents: docs.map(d => ({
          id: d.id,
          title: d.title,
          category: d.category,
          fileType: d.fileType,
          totalLines: d.totalLines,
          totalChunks: d.totalChunks,
          projectId: d.projectId,
          createdAt: d.createdAt,
        })),
      });
    } catch (error) {
      console.error(`  [listDocuments] Error:`, error);
      return JSON.stringify({
        success: false,
        error: `Failed to list documents: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  },
  {
    name: "listDocuments",
    description: "List available documents that have been uploaded. Use this first to see what documents are available before searching. Returns document titles, categories, and line counts.",
    schema: z.object({
      category: z.enum(["course_content", "framework_content"]).optional().describe("Filter by document category"),
      orgId: z.string().optional().describe("Filter by organization ID"),
      projectId: z.string().optional().describe("Filter by project ID"),
      limit: z.number().optional().describe("Maximum number of documents to return (default: 20)"),
    }),
  }
);

/**
 * Semantic search across document chunks using vector similarity.
 */
const searchDocuments = tool(
  async ({ query, category, limit, threshold }: { 
    query: string; 
    category?: string; 
    limit?: number;
    threshold?: number;
  }) => {
    console.log(`  [searchDocuments] Semantic search for: "${query}"`);
    
    try {
      const results = await getDocumentService().searchDocuments({
        query,
        category: category as "course_content" | "framework_content" | undefined,
        limit: limit || 5,
        threshold: threshold || 0.7,
      });

      console.log(`  [searchDocuments] Found ${results.length} matching chunks`);

      return JSON.stringify({
        success: true,
        count: results.length,
        results: results.map(r => ({
          documentId: r.documentId,
          documentTitle: r.documentTitle,
          category: r.category,
          content: r.content,
          chunkIndex: r.chunkIndex,
          startLine: r.startLine,
          endLine: r.endLine,
          similarity: r.similarity.toFixed(3),
        })),
      });
    } catch (error) {
      console.error(`  [searchDocuments] Error:`, error);
      return JSON.stringify({
        success: false,
        error: `Semantic search failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  },
  {
    name: "searchDocuments",
    description: "Semantic search across uploaded documents using AI embeddings. Best for finding conceptually related content, even if exact words don't match. Use for broad topical queries.",
    schema: z.object({
      query: z.string().describe("The search query - describe what you're looking for conceptually"),
      category: z.enum(["course_content", "framework_content"]).optional().describe("Filter by document category"),
      limit: z.number().optional().describe("Maximum number of results (default: 5)"),
      threshold: z.number().optional().describe("Minimum similarity score 0-1 (default: 0.7)"),
    }),
  }
);

/**
 * Full-text search for exact term matches.
 */
const searchDocumentsByText = tool(
  async ({ searchText, category, limit }: { 
    searchText: string; 
    category?: string; 
    limit?: number;
  }) => {
    console.log(`  [searchDocumentsByText] Text search for: "${searchText}"`);
    
    try {
      const results = await getDocumentService().searchDocumentsByText({
        searchText,
        category: category as "course_content" | "framework_content" | undefined,
        limit: limit || 10,
      });

      console.log(`  [searchDocumentsByText] Found ${results.length} matching chunks`);

      return JSON.stringify({
        success: true,
        count: results.length,
        results: results.map(r => ({
          documentId: r.documentId,
          documentTitle: r.documentTitle,
          category: r.category,
          content: r.content,
          chunkIndex: r.chunkIndex,
          startLine: r.startLine,
          endLine: r.endLine,
          rank: r.rank.toFixed(3),
        })),
      });
    } catch (error) {
      console.error(`  [searchDocumentsByText] Error:`, error);
      return JSON.stringify({
        success: false,
        error: `Text search failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  },
  {
    name: "searchDocumentsByText",
    description: "Full-text search for exact term matches in documents. Best for finding specific terms, definitions, names, or exact phrases. More precise than semantic search.",
    schema: z.object({
      searchText: z.string().describe("The exact text or terms to search for"),
      category: z.enum(["course_content", "framework_content"]).optional().describe("Filter by document category"),
      limit: z.number().optional().describe("Maximum number of results (default: 10)"),
    }),
  }
);

/**
 * Get document content by line range.
 */
const getDocumentLines = tool(
  async ({ documentId, documentName, startLine, numLines }: { 
    documentId?: string;
    documentName?: string;
    startLine: number; 
    numLines: number;
  }) => {
    console.log(`  [getDocumentLines] Getting lines ${startLine}-${startLine + numLines - 1}`);
    
    try {
      let docId = documentId;
      
      // If documentName provided, look up the document first
      if (!docId && documentName) {
        const doc = await getDocumentService().getDocumentByName(documentName);
        if (!doc) {
          return JSON.stringify({
            success: false,
            error: `Document not found: ${documentName}`,
          });
        }
        docId = doc.id;
      }
      
      if (!docId) {
        return JSON.stringify({
          success: false,
          error: "Either documentId or documentName must be provided",
        });
      }

      const results = await getDocumentService().getDocumentLines({
        documentId: docId,
        startLine,
        numLines,
      });

      console.log(`  [getDocumentLines] Retrieved ${results.length} chunks covering requested lines`);

      // Combine content from relevant chunks
      const content = results.map(r => r.content).join("\n");

      return JSON.stringify({
        success: true,
        documentId: docId,
        startLine,
        numLines,
        chunksReturned: results.length,
        content,
      });
    } catch (error) {
      console.error(`  [getDocumentLines] Error:`, error);
      return JSON.stringify({
        success: false,
        error: `Failed to get document lines: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  },
  {
    name: "getDocumentLines",
    description: "Retrieve specific lines from a document. Use after search results indicate relevant content at specific line numbers. Good for getting more context around a search result.",
    schema: z.object({
      documentId: z.string().optional().describe("The document ID (from search results or listDocuments)"),
      documentName: z.string().optional().describe("The document title/name to look up (alternative to documentId)"),
      startLine: z.number().describe("The line number to start from"),
      numLines: z.number().describe("Number of lines to retrieve"),
    }),
  }
);

/**
 * Get full document content by name.
 */
const getDocumentByName = tool(
  async ({ documentName, section }: { 
    documentName: string;
    section?: string;
  }) => {
    console.log(`  [getDocumentByName] Getting document: "${documentName}"${section ? `, section: ${section}` : ''}`);
    
    try {
      const doc = await getDocumentService().getDocumentByName(documentName);
      
      if (!doc) {
        return JSON.stringify({
          success: false,
          error: `Document not found: ${documentName}`,
        });
      }

      // Get full content
      const content = await getDocumentService().getDocumentContent(doc.id);

      // If section specified, try to extract it (basic heading-based extraction)
      let resultContent = content;
      if (section) {
        const sectionRegex = new RegExp(`(?:^|\\n)(#{1,3}\\s*${section}[^\\n]*)([\\s\\S]*?)(?=\\n#{1,3}\\s|$)`, 'i');
        const match = content.match(sectionRegex);
        if (match) {
          resultContent = match[1] + match[2];
        } else {
          // Try to find section by line contains
          const lines = content.split('\n');
          const startIdx = lines.findIndex(l => l.toLowerCase().includes(section.toLowerCase()));
          if (startIdx !== -1) {
            // Return from section start to next major heading or 100 lines
            const endIdx = Math.min(startIdx + 100, lines.length);
            resultContent = lines.slice(startIdx, endIdx).join('\n');
          }
        }
      }

      console.log(`  [getDocumentByName] Retrieved document with ${resultContent.length} chars`);

      // Warn if content is very large
      const isLarge = resultContent.length > 10000;

      return JSON.stringify({
        success: true,
        document: {
          id: doc.id,
          title: doc.title,
          category: doc.category,
          fileType: doc.fileType,
          totalLines: doc.totalLines,
        },
        contentLength: resultContent.length,
        warning: isLarge ? "Large document - consider using searchDocuments or getDocumentLines for specific sections" : undefined,
        content: isLarge ? resultContent.substring(0, 10000) + "\n\n[TRUNCATED - use getDocumentLines for specific sections]" : resultContent,
      });
    } catch (error) {
      console.error(`  [getDocumentByName] Error:`, error);
      return JSON.stringify({
        success: false,
        error: `Failed to get document: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });
    }
  },
  {
    name: "getDocumentByName",
    description: "Retrieve a document's full content by name. WARNING: Only use for small documents or when full context is truly needed. For large documents, prefer searchDocuments or getDocumentLines to manage context.",
    schema: z.object({
      documentName: z.string().describe("The document title or filename to retrieve"),
      section: z.string().optional().describe("Optional: specific section heading to extract"),
    }),
  }
);

/**
 * Backend tools that are executed server-side (not frontend CopilotKit tools).
 */
const backendTools = [webSearch, listDocuments, searchDocuments, searchDocumentsByText, getDocumentLines, getDocumentByName];

// ============================================================================
// HELPER: Trim messages to prevent state explosion
// ============================================================================

/**
 * Trims message history to keep only recent messages.
 * This prevents the state from growing unbounded and causing
 * JSON serialization errors in FileSystemPersistence.
 * 
 * Strategy:
 * - Always keep system messages (contain important instructions)
 * - Keep only the most recent N conversation messages
 * - Ensure tool_use/tool_result pairs stay together (Anthropic requirement)
 * - This allows long-running sessions without hitting string length limits
 */
function trimMessages(messages: BaseMessage[], keepRecent: number = 40): BaseMessage[] {
  if (messages.length <= keepRecent) {
    return messages;
  }
  
  // Always keep system messages
  const systemMessages = messages.filter(m => {
    const msgType = (m as any)._getType?.() || (m as any).constructor?.name || '';
    return msgType === 'system' || msgType === 'SystemMessage';
  });
  
  // Get non-system messages
  const otherMessages = messages.filter(m => {
    const msgType = (m as any)._getType?.() || (m as any).constructor?.name || '';
    return msgType !== 'system' && msgType !== 'SystemMessage';
  });
  
  // Find a safe cut point that doesn't break tool_use/tool_result pairs
  // We need to ensure we don't start with a tool_result message
  let startIdx = Math.max(0, otherMessages.length - keepRecent);
  
  // Scan forward to find a safe starting point (human message is always safe)
  while (startIdx < otherMessages.length) {
    const msg = otherMessages[startIdx];
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
    
    // Safe to start at: human message, or AI message without pending tool calls
    if (msgType === 'human' || msgType === 'HumanMessage') {
      break;
    }
    
    // Tool messages need their corresponding AI tool_use message
    if (msgType === 'tool' || msgType === 'ToolMessage') {
      startIdx++;
      continue;
    }
    
    // AI messages are safe if they don't have tool calls that need results
    if (msgType === 'ai' || msgType === 'AIMessage' || msgType === 'AIMessageChunk') {
      const aiMsg = msg as any;
      const hasToolCalls = aiMsg.tool_calls?.length > 0 || 
                          (aiMsg.additional_kwargs?.tool_calls?.length > 0);
      if (!hasToolCalls) {
        break;
      }
      // This AI has tool calls - check if the next message is the tool result
      const nextIdx = startIdx + 1;
      if (nextIdx < otherMessages.length) {
        const nextMsg = otherMessages[nextIdx];
        const nextType = (nextMsg as any)._getType?.() || (nextMsg as any).constructor?.name || '';
        if (nextType === 'tool' || nextType === 'ToolMessage') {
          // We'd be cutting in the middle of a tool call - skip both
          startIdx++;
          continue;
        }
      }
      break;
    }
    
    // Default: try the next message
    startIdx++;
  }
  
  const recentMessages = otherMessages.slice(startIdx);
  
  console.log(`  [TRIM] Messages reduced: ${messages.length} -> ${systemMessages.length + recentMessages.length} (${systemMessages.length} system + ${recentMessages.length} recent)`);
  
  return [...systemMessages, ...recentMessages];
}

// ============================================================================
// HELPER: Extract created nodes from message history
// ============================================================================

/**
 * Parses tool result messages to extract successfully created nodes.
 * This allows us to track what has been created across conversation turns.
 */
function extractCreatedNodesFromMessages(messages: any[]): CreatedNode[] {
  const createdNodes: CreatedNode[] = [];
  
  for (const msg of messages) {
    // Tool messages contain results from createNode calls
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
    if (msgType === 'tool' || msgType === 'ToolMessage') {
      try {
        const content = typeof msg.content === 'string' 
          ? msg.content 
          : JSON.stringify(msg.content);
        
        // Parse the tool result
        const result = typeof msg.content === 'string' 
          ? JSON.parse(msg.content) 
          : msg.content;
        
        // Check if this is a successful createNode result
        if (result?.success === true && result?.newNodeId && result?.message) {
          // Extract title from message: "Successfully created \"Title\" (Template) under \"Parent\"."
          const titleMatch = result.message.match(/Successfully created "([^"]+)"/);
          const parentMatch = result.message.match(/under "([^"]+)"/);
          
          if (titleMatch) {
            createdNodes.push({
              parentNodeId: null, // We don't have this easily from the message
              title: titleMatch[1],
              nodeId: result.newNodeId,
              templateName: result.templateName || result.nodeType || 'unknown',
            });
          }
        }
      } catch (e) {
        // Ignore parsing errors for non-createNode results
      }
    }
  }
  
  return createdNodes;
}

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const SYSTEM_PROMPT = `You are a Node Creation Expert for BlueprintCMS. Your role is to help users create well-structured content hierarchies by thoughtfully planning and then creating nodes.

## Your Capabilities

You have access to frontend tools that allow you to:
1. **getProjectHierarchyInfo** - Understand the project's hierarchy structure, level names, and coding configurations
2. **getAvailableTemplates** - See what node templates are available for the current context
3. **getNodeChildren** - Check existing children of a node
4. **createNode** - Create new nodes in the hierarchy
5. **requestEditMode** - Request edit access if needed
6. **releaseEditMode** - Release edit access when done

**Human-in-the-Loop Approval Tools:**
7. **requestPlanApproval** - ALWAYS use this when proposing to create nodes. Shows an interactive approval popup to the user. DO NOT ask "shall I proceed?" - just call this tool directly.
8. **requestActionApproval** - Request confirmation for destructive/sensitive actions (like deleting nodes)
9. **offerOptions** - Present the user with choices and wait for their selection

You also have access to backend tools:
10. **web_search** - Search the web for current information using Perplexity AI. Use this to research topics, find recent events, look up facts, or gather information before creating content nodes. Returns an answer with citations.

**Document Search Tools (for uploaded Course Content and Framework Content):**
11. **listDocuments** - List all available uploaded documents. Use FIRST to see what's available.
12. **searchDocuments** - Semantic search using AI embeddings. Best for conceptual/topical queries.
13. **searchDocumentsByText** - Full-text search for exact terms, definitions, or specific phrases.
14. **getDocumentLines** - Retrieve specific line ranges from a document. Use after search to get more context.
15. **getDocumentByName** - Get full document content. WARNING: Only use for small documents.

## CRITICAL: Plan-First Approach with Automatic Approval Request

**You MUST follow a strict two-phase approach:**

### PHASE 1: PLANNING & APPROVAL REQUEST

When the user requests node creation:

1. **Gather Context First**
   - Call getProjectHierarchyInfo to understand hierarchy levels
   - Call getAvailableTemplates to see available templates
   - Review EXISTING_NODES context to see what already exists

2. **Create and Validate the Plan**
   - Each node title must be UNIQUE under its parent
   - Cross-check against EXISTING_NODES - skip any that already exist
   - Cross-check against ALREADY_CREATED_NODES - never duplicate

3. **IMMEDIATELY Submit for Approval** (DO NOT ask "shall I proceed?")
   
   **IMPORTANT:** After creating your plan, ALWAYS call requestPlanApproval immediately. 
   DO NOT output a plan and then ask the user if they're ready - the approval tool IS the mechanism for user confirmation.
   
   Call requestPlanApproval with:
   - planSummary: Brief description (e.g., "Create 4 nodes: 1 module with 3 lessons")
   - nodeCount: The total number of nodes to create
   - proposedStructure: Text tree representation showing the hierarchy
   - actionType: "Create Course Structure" or similar

4. **Wait for the user's response via the approval popup:**
   - If APPROVED: Proceed to Phase 2 (Execution)
   - If REJECTED: Ask what changes they'd like, then revise the plan and submit for approval again

### PHASE 2: EXECUTION (Only after approval)

Only after you receive "APPROVED" from the requestPlanApproval tool:

1. Request edit mode if needed
2. Create nodes ONE AT A TIME in order:
   - Parents before children (top-down)
   - Wait for each node to succeed before the next
3. Track what you've created - NEVER recreate
4. Report completion when done

## CRITICAL: Preventing Duplicates

**NEVER create a node that already exists or that you've already created.**

Before EACH createNode call, check:
1. **EXISTING_NODES context** - nodes already in the project
2. **ALREADY_CREATED_NODES list** - nodes you created this session
3. **Tool results** - "Successfully created X" means X exists

If a title appears in any of these lists under the same parent: **SKIP IT**.

## Example Workflow

**User:** "Create a module about React Hooks with lessons on useState, useEffect, and custom hooks"

**You (Phase 1 - Planning & Approval):**
1. Call getProjectHierarchyInfo → Learn hierarchy has Level 2 (Modules) and Level 3 (Lessons)
2. Call getAvailableTemplates → Get template IDs
3. Check EXISTING_NODES → No "React Hooks" exists
4. **IMMEDIATELY** call requestPlanApproval with:
   - planSummary: "Create 4 nodes: 1 React Hooks module with 3 lesson nodes (useState, useEffect, Custom Hooks)"
   - nodeCount: 4
   - proposedStructure: "React Hooks\\n├── useState Hook\\n├── useEffect Hook\\n└── Custom Hooks"
   - actionType: "Create Course Structure"

**User sees approval popup and clicks "Approve"**

**You (Phase 2 - Execution):**
5. requestEditMode
6. createNode "React Hooks" → Success, ID: abc123
7. createNode "useState Hook" under abc123 → Success
8. createNode "useEffect Hook" under abc123 → Success
9. createNode "Custom Hooks" under abc123 → Success
10. Report: "Created 4 nodes successfully!"

## Important Rules

- **NEVER ASK "SHALL I PROCEED?"** - Use requestPlanApproval tool instead, it provides the approval UI
- **ALWAYS USE APPROVAL TOOL** - When proposing ANY node creation, call requestPlanApproval immediately
- **WAIT FOR APPROVAL** - Do NOT create nodes until you receive "APPROVED" response from the tool
- **ONE AT A TIME** - Create nodes sequentially, not all at once
- **PARENTS FIRST** - Create parent nodes before their children
- **NO DUPLICATES** - Check all sources before each create
- **TRACK PROGRESS** - Remember what you've created
- **RESEARCH WHEN NEEDED** - Use web_search to gather information for content planning
- **HANDLE REJECTION** - If rejected, ask what changes the user wants and revise your plan

## When to Use Web Search

Use the **web_search** tool when:
- The user asks about current events, recent developments, or up-to-date information
- You need to research a topic before creating content nodes
- The user asks factual questions that require external knowledge
- You want to find best practices or standards for content structure
- You need to verify information or get accurate details

The web_search tool returns:
- An **answer** synthesizing information from the web
- **citations** with source URLs you can reference

Example: "Create a module about the latest React 19 features"
1. First use web_search to research "React 19 new features and changes"
2. Use the search results to plan accurate, up-to-date content
3. Then proceed with the normal planning and creation workflow

## Document Search Strategy (Context Management)

When you need information from uploaded documents, follow this priority to manage context efficiently:

1. **Start with listDocuments** - First check what documents are available
2. **Use searchDocuments (semantic)** - For conceptual queries, finding related content, broad topics
3. **Use searchDocumentsByText (exact)** - For specific terms, definitions, names, or exact phrases
4. **Use getDocumentLines** - To retrieve specific portions once you know the location from search results
5. **Use getDocumentByName** - ONLY for small documents or when full context is absolutely needed

**IMPORTANT: Context Management Rules**
- AVOID loading full documents unless necessary - they consume context and reduce response quality
- Use targeted searches to find relevant sections first
- When search results show relevant content at specific lines, use getDocumentLines to expand context
- Prefer semantic search first to find relevant topics, then refine with text search for specifics
- Document categories: "course_content" for training materials, "framework_content" for competency frameworks

**Example Document Workflow:**
1. User asks: "What does the framework say about leadership competencies?"
2. You call listDocuments to see available framework documents
3. You call searchDocuments with query "leadership competencies"
4. Results show relevant chunks with line numbers and similarity scores
5. If you need more context, call getDocumentLines with the document ID and line range
6. Use the retrieved content to answer the user's question

Remember: Quality over speed. A well-planned structure is worth the extra step.`;

// ============================================================================
// CHAT NODE
// ============================================================================

/**
 * Main chat node that processes messages and calls tools.
 * Binds frontend tools from CopilotKit state to the model.
 * Tracks created nodes to prevent duplicates.
 */
async function chat_node(
  state: NodeExpertState,
  config: RunnableConfig
): Promise<Partial<NodeExpertState>> {
  console.log('\n[node_expert] ============ chat_node called ============');
  console.log('  State keys:', Object.keys(state));
  console.log('  Has copilotkit?', !!state.copilotkit);
  console.log('  Has actions?', !!state.copilotkit?.actions);
  console.log('  Actions count:', state.copilotkit?.actions?.length ?? 0);
  console.log('  Messages count:', state.messages?.length ?? 0);
  
  // Extract created nodes from message history
  const createdNodesFromMessages = extractCreatedNodesFromMessages(state.messages || []);
  
  // Merge with any existing tracked nodes in state
  const existingCreatedNodes = state.createdNodes || [];
  const allCreatedNodes = [...existingCreatedNodes];
  for (const node of createdNodesFromMessages) {
    if (!allCreatedNodes.some(n => n.nodeId === node.nodeId)) {
      allCreatedNodes.push(node);
    }
  }
  
  console.log('  Created nodes tracked:', allCreatedNodes.length);
  if (allCreatedNodes.length > 0) {
    console.log('\n  [ALREADY CREATED NODES] ============================');
    for (const node of allCreatedNodes) {
      console.log(`    - "${node.title}" (${node.templateName}) [${node.nodeId.substring(0, 8)}...]`);
    }
    console.log('  [/ALREADY CREATED NODES] ===========================\n');
  }
  
  // Log the last few messages for context
  if (state.messages?.length > 0) {
    const recentMessages = state.messages.slice(-3);
    console.log('\n  [RECENT MESSAGES] ================================');
    for (const msg of recentMessages) {
      const role = (msg as any)._getType?.() || (msg as any).constructor?.name || 'unknown';
      let content = '';
      if (typeof (msg as any).content === 'string') {
        content = (msg as any).content.substring(0, 150);
      } else if (Array.isArray((msg as any).content)) {
        const textBlock = (msg as any).content.find((b: any) => b.type === 'text');
        content = textBlock?.text?.substring(0, 150) || '[complex content]';
      }
      console.log(`  [${role}]: ${content}${content.length >= 150 ? '...' : ''}`);
    }
    console.log('  [/RECENT MESSAGES] ===============================\n');
  }
  
  // Access frontend tools from CopilotKit state
  const frontendTools = state.copilotkit?.actions ?? [];
  
  console.log('  Frontend tools:', frontendTools.map((t: any) => t.name).join(', '));
  console.log('  Backend tools:', backendTools.map(t => t.name).join(', '));

  // Combine frontend and backend tools
  const allTools = [...frontendTools, ...backendTools];

  // Bind all tools to the model
  const modelWithTools = allTools.length > 0 
    ? model.bindTools(allTools)
    : model;

  // Build dynamic system message with created nodes context
  let systemContent = SYSTEM_PROMPT;
  
  // Append created nodes list to help model avoid duplicates
  if (allCreatedNodes.length > 0) {
    const nodeList = allCreatedNodes
      .map(n => `- "${n.title}" (${n.templateName}) [ID: ${n.nodeId.substring(0, 8)}...]`)
      .join('\n');
    systemContent += `\n\n## ALREADY_CREATED_NODES (DO NOT RECREATE THESE)

**EXECUTION IN PROGRESS** - You have already created ${allCreatedNodes.length} node(s) in this session:

${nodeList}

**Rules:**
- DO NOT call createNode for any title that appears above
- If your plan included a node that's now listed here, it's DONE - move to the next
- Continue creating remaining nodes from your plan
- When all planned nodes are created, report completion to the user`;
  } else {
    systemContent += `\n\n## SESSION STATUS

No nodes created yet. If the user is asking you to create content:
1. First gather context (getProjectHierarchyInfo, getAvailableTemplates)
2. Output your complete plan with all proposed nodes
3. Only after planning, begin execution`;
  }

  // Create system message with dynamic content
  const systemMessage = new SystemMessage({
    content: systemContent,
  });

  console.log('  Invoking model with', allTools.length, 'tools (', frontendTools.length, 'frontend,', backendTools.length, 'backend)...');

  // Filter out orphaned tool results (keep thinking blocks since thinking is enabled)
  const filteredMessages = filterOrphanedToolResults(state.messages || []);
  console.log(`  Messages: ${state.messages?.length ?? 0} -> ${filteredMessages.length} after filtering`);

  // Trim messages to prevent state explosion (keeps state size under control)
  const trimmedMessages = trimMessages(filteredMessages, 40);

  // Customize config to emit tool calls to the frontend for rendering
  // Include ALL frontend tools so their execution is visible in the chat
  const customConfig = copilotkitCustomizeConfig(config, {
    emitToolCalls: true // Emit ALL tool calls for rendering
  });

  // Invoke the model with system prompt and trimmed chat history
  const response = await modelWithTools.invoke(
    [systemMessage, ...trimmedMessages],
    customConfig
  );

  console.log('  Model response received');
  
  const aiResponse = response as AIMessage;
  
  // Log tool calls with full arguments
  console.log('  Has tool calls?', !!aiResponse.tool_calls?.length);
  if (aiResponse.tool_calls?.length) {
    console.log('\n  [TOOL CALLS] =====================================');
    for (const tc of aiResponse.tool_calls) {
      console.log(`  Tool: ${tc.name}`);
      console.log(`  ID: ${tc.id}`);
      console.log(`  Arguments: ${JSON.stringify(tc.args, null, 4).split('\n').join('\n  ')}`);
      
      // Warn if this looks like a duplicate
      if (tc.name === 'createNode' && tc.args?.title) {
        const isDuplicate = allCreatedNodes.some(
          n => n.title.toLowerCase() === tc.args.title.toLowerCase()
        );
        if (isDuplicate) {
          console.log(`  ** WARNING: "${tc.args.title}" may be a DUPLICATE! **`);
        }
      }
      console.log('  ---');
    }
    console.log('  [/TOOL CALLS] ====================================\n');
  }

  return {
    messages: [response],
    createdNodes: allCreatedNodes,
  };
}

// ============================================================================
// ROUTING LOGIC
// ============================================================================

/**
 * Names of backend tools that are executed server-side.
 */
const backendToolNames = new Set(backendTools.map(t => t.name));

/**
 * Determines the next node to execute.
 *
 * Returns:
 * - "__end__" when: No tool calls, or tool calls are only frontend actions
 * - "execute_tools" when: Any backend tool exists (executed first, even if mixed)
 *
 * IMPORTANT: When there are mixed tool calls (both frontend and backend),
 * we MUST execute backend tools first. After backend execution, 
 * afterToolExecution() will route to __end__ for frontend tools.
 */
function shouldContinue(state: NodeExpertState): "__end__" | "execute_tools" {
  console.log('\n[node_expert] shouldContinue called');
  
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  console.log('  Last message type:', lastMessage?.constructor?.name);
  console.log('  Tool calls count:', lastMessage.tool_calls?.length ?? 0);

  // If no tool calls, we're done
  if (!lastMessage.tool_calls?.length) {
    console.log('  -> Routing to __end__ (no tool calls)');
    return "__end__";
  }

  // Get the frontend actions from CopilotKit
  const actions = state.copilotkit?.actions ?? [];
  const frontendActionNames = new Set(actions.map((action: { name: string }) => action.name));

  console.log('  Frontend actions:', Array.from(frontendActionNames).join(', ') || '(none)');
  console.log('  Backend tools:', Array.from(backendToolNames).join(', '));
  console.log('  Tool calls:', lastMessage.tool_calls.map(tc => tc.name).join(', '));

  // Check ALL tool calls to determine if we have backend and/or frontend tools
  const hasBackendTools = lastMessage.tool_calls.some(tc => backendToolNames.has(tc.name));
  const hasFrontendTools = lastMessage.tool_calls.some(tc => frontendActionNames.has(tc.name));

  console.log('  Has backend tools:', hasBackendTools);
  console.log('  Has frontend tools:', hasFrontendTools);

  // CRITICAL: If ANY backend tools exist, execute them FIRST
  // afterToolExecution() will then route to __end__ for pending frontend tools
  if (hasBackendTools) {
    if (hasFrontendTools) {
      console.log('  -> MIXED TOOLS DETECTED - executing backend tools first');
    }
    const backendToolCalls = lastMessage.tool_calls.filter(tc => backendToolNames.has(tc.name));
    console.log('  -> Routing to execute_tools (backend tools:', backendToolCalls.map(tc => tc.name).join(', '), ')');
    return "execute_tools";
  }
  
  // Only frontend tools - route to __end__ for CopilotKit execution
  if (hasFrontendTools) {
    console.log('  -> Routing to __end__ (frontend tools only)');
    return "__end__";
  }

  // No matching tools found - route to __end__
  console.log('  -> Routing to __end__ (no matching tools)');
  return "__end__";
}

/**
 * Determines routing after backend tool execution.
 *
 * Returns:
 * - "__end__" when: There are pending frontend tools that need CopilotKit execution
 * - "chat_node" when: All tools have been executed, continue conversation
 *
 * This handles the case where the AI made mixed tool calls (frontend + backend).
 * Backend tools were just executed, now we check if frontend tools are pending.
 */
function afterToolExecution(state: NodeExpertState): "__end__" | "chat_node" {
  console.log('\n[node_expert] afterToolExecution called');
  
  const messages = state.messages;
  
  // Find the last AI message that has tool calls
  let aiMessageIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
    if (msgType === 'ai' || msgType === 'AIMessage' || msgType === 'AIMessageChunk') {
      const aiMsg = msg as AIMessage;
      if (aiMsg.tool_calls?.length) {
        aiMessageIndex = i;
        break;
      }
    }
  }
  
  if (aiMessageIndex === -1) {
    console.log('  No AI message with tool calls found, routing to chat_node');
    return "chat_node";
  }
  
  const aiMessage = messages[aiMessageIndex] as AIMessage;
  const toolCallIds = new Set(aiMessage.tool_calls!.map(tc => tc.id));
  
  // Find which tool calls have results
  const toolResultIds = new Set<string>();
  for (let i = aiMessageIndex + 1; i < messages.length; i++) {
    const msg = messages[i];
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || '';
    if (msgType === 'tool' || msgType === 'ToolMessage') {
      const toolMsg = msg as ToolMessage;
      toolResultIds.add(toolMsg.tool_call_id);
    }
  }
  
  // Find pending tool calls (those without results)
  const pendingToolCalls = aiMessage.tool_calls!.filter(tc => !toolResultIds.has(tc.id!));
  
  console.log('  Total tool calls:', aiMessage.tool_calls!.length);
  console.log('  Tool results received:', toolResultIds.size);
  console.log('  Pending tool calls:', pendingToolCalls.length);
  
  if (pendingToolCalls.length > 0) {
    // Check if any pending are frontend tools
    const actions = state.copilotkit?.actions ?? [];
    const frontendActionNames = new Set(actions.map((action: { name: string }) => action.name));
    const pendingFrontendTools = pendingToolCalls.filter(tc => frontendActionNames.has(tc.name));
    
    console.log('  Pending frontend tools:', pendingFrontendTools.map(tc => tc.name).join(', ') || '(none)');
    
    if (pendingFrontendTools.length > 0) {
      console.log('  -> Routing to __end__ (pending frontend tools need CopilotKit execution)');
      return "__end__";
    }
  }
  
  console.log('  -> Routing to chat_node (all tools executed)');
  return "chat_node";
}

// ============================================================================
// TOOL EXECUTION NODE
// ============================================================================

/**
 * ToolNode for executing backend tools (like web_search).
 * Frontend tools are NOT executed here - they route to END for CopilotKit.
 */
const toolNode = new ToolNode(backendTools);

// ============================================================================
// GRAPH DEFINITION
// ============================================================================

/**
 * Build the LangGraph workflow.
 *
 * Structure with mixed tool support:
 * START -> chat_node -> (shouldContinue) -> execute_tools -> (afterToolExecution) -> chat_node
 *                    |                                    |
 *                    -> (frontend only) -> END            -> (pending frontend) -> END
 *
 * - chat_node: Processes messages and generates tool calls
 * - execute_tools: Executes backend tools (web_search, document tools)
 * - afterToolExecution: Routes to __end__ if frontend tools are pending
 * - Frontend tools route to END for CopilotKit client-side execution
 *
 * MIXED TOOL HANDLING:
 * When AI calls both backend and frontend tools in one message:
 * 1. shouldContinue routes to execute_tools (backend first)
 * 2. Backend tools execute, results added to messages
 * 3. afterToolExecution detects pending frontend tools
 * 4. Routes to __end__ for CopilotKit to execute frontend tools
 * 5. Conversation resumes with all tool results
 */
const workflow = new StateGraph(NodeExpertStateAnnotation)
  .addNode("chat_node", chat_node)
  .addNode("execute_tools", toolNode)
  .addEdge(START, "chat_node")
  .addConditionalEdges("chat_node", shouldContinue, {
    execute_tools: "execute_tools",
    __end__: END,
  })
  .addConditionalEdges("execute_tools", afterToolExecution, {
    chat_node: "chat_node",
    __end__: END,
  });

// ============================================================================
// PERSISTENCE SETUP - PostgreSQL via Supabase
// ============================================================================

/**
 * Initialize PostgresSaver for persistent conversation storage.
 * Connects to local Supabase PostgreSQL database.
 * 
 * Environment variable SUPABASE_DB_URL should be set to:
 * postgresql://postgres:postgres@localhost:15322/postgres
 */
const SUPABASE_DB_URL = process.env.SUPABASE_DB_URL || "postgresql://postgres:postgres@localhost:15322/postgres";

console.log('[node_expert] Initializing PostgreSQL checkpointer...');
console.log('[node_expert] DB URL:', SUPABASE_DB_URL.replace(/:[^:@]+@/, ':****@')); // Hide password

const checkpointer = PostgresSaver.fromConnString(SUPABASE_DB_URL);

// Setup creates necessary tables if they don't exist
// This is idempotent and safe to call on every startup
await checkpointer.setup();
console.log('[node_expert] PostgreSQL checkpointer initialized successfully');

// Compile the graph with PostgreSQL persistence
export const agent = workflow.compile({
  checkpointer,
});

