/**
 * Researcher Agent
 *
 * Deep knowledge gathering on industry, topics, regulations, and personas.
 * Uses web search, document RAG, and media library to build comprehensive research briefs.
 *
 * Tools:
 * - web_search - Perplexity API search
 * - listDocuments, searchDocuments, searchDocumentsByText, getDocumentLines, getDocumentByName - Document RAG
 * - searchMicroverse, getMicroverseDetails - Media library search
 *
 * Input: Reads projectBrief from state
 * Output: researchFindings in shared state
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { AIMessage, SystemMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { ChatAnthropic } from "@langchain/anthropic";
import { z } from "zod";
import OpenAI from "openai";
import { createDocumentService, DocumentService } from "../lib/documents";
import type { OrchestratorState, ResearchBrief } from "../state/agent-state";
import { getCondensedBrief } from "../state/agent-state";

// Centralized context management utilities
import {
  filterOrphanedToolResults,
  clearOldToolResults,
  hasUsableResponse,
  MESSAGE_LIMITS,
} from "../utils";

// Message filtering now handled by centralized utils/context-management.ts

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

const researcherModel = new ChatAnthropic({
  model: "claude-opus-4-5-20251101",
  maxTokens: 16000,
  temperature: 0.7,
});

// Empty response detection now handled by centralized utils/context-management.ts

// ============================================================================
// PERPLEXITY WEB SEARCH TOOL
// ============================================================================

const perplexityClient = new OpenAI({
  apiKey: process.env.PERPLEXITY_API_KEY,
  baseURL: "https://api.perplexity.ai",
});

const webSearch = tool(
  async ({ query }: { query: string }) => {
    console.log(`  [researcher/web_search] Searching: "${query}"`);

    try {
      const response = await perplexityClient.chat.completions.create({
        model: "sonar-pro",
        messages: [{ role: "user", content: query }],
      });

      const content = response.choices[0]?.message?.content || "No results found.";
      const citations = (response as any).citations || [];

      console.log(`  [researcher/web_search] Got ${citations.length} citations`);

      return JSON.stringify({
        answer: content,
        citations: citations,
        query: query,
      });
    } catch (error) {
      console.error(`  [researcher/web_search] Error:`, error);
      return JSON.stringify({
        error: `Web search failed: ${error instanceof Error ? error.message : "Unknown error"}`,
        query: query,
      });
    }
  },
  {
    name: "web_search",
    description:
      "Search the web for current information using Perplexity AI. Use for researching industry topics, regulations, best practices, and current trends.",
    schema: z.object({
      query: z.string().describe("The search query to look up"),
    }),
  }
);

// ============================================================================
// DOCUMENT RAG TOOLS
// ============================================================================

let documentService: DocumentService | null = null;

function getDocumentService(): DocumentService {
  if (!documentService) {
    documentService = createDocumentService();
  }
  return documentService;
}

const listDocuments = tool(
  async ({ category, limit }: { category?: string; limit?: number }) => {
    console.log(`  [researcher/listDocuments] Listing - category: ${category || "all"}`);

    try {
      const docs = await getDocumentService().listDocuments({
        category: category as "course_content" | "framework_content" | undefined,
        limit: limit || 20,
      });

      console.log(`  [researcher/listDocuments] Found ${docs.length} documents`);

      return JSON.stringify({
        success: true,
        count: docs.length,
        documents: docs.map((d) => ({
          id: d.id,
          title: d.title,
          category: d.category,
          fileType: d.fileType,
          totalLines: d.totalLines,
        })),
      });
    } catch (error) {
      return JSON.stringify({
        success: false,
        error: `Failed to list documents: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  },
  {
    name: "listDocuments",
    description: "List available uploaded documents. Use first to see what's available before searching.",
    schema: z.object({
      category: z.enum(["course_content", "framework_content"]).optional(),
      limit: z.coerce.number().optional().describe("Max documents to return (default: 20)"),
    }),
  }
);

const searchDocuments = tool(
  async ({ query, category, limit, threshold }: { query: string; category?: string; limit?: number; threshold?: number }) => {
    console.log(`  [researcher/searchDocuments] Hybrid search: "${query}"`);

    try {
      const { source, results } = await getDocumentService().hybridSearch({
        query,
        category: category as "course_content" | "framework_content" | undefined,
        limit: limit || 5,
        threshold: threshold,
      });

      console.log(`  [researcher/searchDocuments] Found ${results.length} chunks via ${source} search`);

      return JSON.stringify({
        success: true,
        searchType: source,
        count: results.length,
        results: results.map((r) => ({
          documentId: r.documentId,
          documentTitle: r.documentTitle,
          category: r.category,
          content: r.content,
          chunkIndex: r.chunkIndex,
          startLine: r.startLine,
          endLine: r.endLine,
          similarity: r.similarity?.toFixed(3),
          rank: r.rank?.toFixed(3),
        })),
      });
    } catch (error) {
      return JSON.stringify({
        success: false,
        error: `Search failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  },
  {
    name: "searchDocuments",
    description: "Search documents using hybrid semantic + text search. Tries AI semantic matching first, falls back to keyword search if no results. Best for longer, descriptive queries (e.g., 'how to perform pre-trip inspections' rather than single words).",
    schema: z.object({
      query: z.string().describe("Descriptive search query - longer phrases work better than single words"),
      category: z.enum(["course_content", "framework_content"]).optional(),
      limit: z.coerce.number().optional().describe("Max results (default: 5)"),
      threshold: z.coerce.number().optional().describe("Min similarity 0-1 (default: 0.5)"),
    }),
  }
);

const searchDocumentsByText = tool(
  async ({ searchText, category, limit }: { searchText: string; category?: string; limit?: number }) => {
    console.log(`  [researcher/searchDocumentsByText] Text search: "${searchText}"`);

    try {
      const results = await getDocumentService().searchDocumentsByText({
        searchText,
        category: category as "course_content" | "framework_content" | undefined,
        limit: limit || 10,
      });

      console.log(`  [researcher/searchDocumentsByText] Found ${results.length} chunks`);

      return JSON.stringify({
        success: true,
        count: results.length,
        results: results.map((r) => ({
          documentId: r.documentId,
          documentTitle: r.documentTitle,
          content: r.content,
          startLine: r.startLine,
          endLine: r.endLine,
          rank: r.rank.toFixed(3),
        })),
      });
    } catch (error) {
      return JSON.stringify({
        success: false,
        error: `Text search failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  },
  {
    name: "searchDocumentsByText",
    description: "Full-text keyword search. Best for single words, exact terms, acronyms (e.g., 'GVM', 'HVNL'), and specific phrases. Use this when searchDocuments returns no results for short queries.",
    schema: z.object({
      searchText: z.string().describe("Keyword or exact phrase to search for - good for single words and acronyms"),
      category: z.enum(["course_content", "framework_content"]).optional(),
      limit: z.coerce.number().optional().describe("Max results (default: 10)"),
    }),
  }
);

const getDocumentLines = tool(
  async ({ documentId, documentName, startLine, numLines }: { documentId?: string; documentName?: string; startLine: number; numLines: number }) => {
    console.log(`  [researcher/getDocumentLines] Getting lines ${startLine}-${startLine + numLines - 1}`);

    try {
      let docId = documentId;

      if (!docId && documentName) {
        const doc = await getDocumentService().getDocumentByName(documentName);
        if (!doc) {
          return JSON.stringify({ success: false, error: `Document not found: ${documentName}` });
        }
        docId = doc.id;
      }

      if (!docId) {
        return JSON.stringify({ success: false, error: "Either documentId or documentName required" });
      }

      const results = await getDocumentService().getDocumentLines({
        documentId: docId,
        startLine,
        numLines,
      });

      const content = results.map((r) => r.content).join("\n");

      return JSON.stringify({
        success: true,
        documentId: docId,
        startLine,
        numLines,
        content,
      });
    } catch (error) {
      return JSON.stringify({
        success: false,
        error: `Failed to get lines: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  },
  {
    name: "getDocumentLines",
    description: "Retrieve specific lines from a document. Use after search to get more context.",
    schema: z.object({
      documentId: z.string().optional(),
      documentName: z.string().optional(),
      startLine: z.coerce.number().describe("Line number to start from"),
      numLines: z.coerce.number().describe("Number of lines to retrieve"),
    }),
  }
);

const getDocumentByName = tool(
  async ({ documentName, section }: { documentName: string; section?: string }) => {
    console.log(`  [researcher/getDocumentByName] Getting: "${documentName}"`);

    try {
      const doc = await getDocumentService().getDocumentByName(documentName);

      if (!doc) {
        return JSON.stringify({ success: false, error: `Document not found: ${documentName}` });
      }

      const content = await getDocumentService().getDocumentContent(doc.id);

      let resultContent = content;
      if (section) {
        const sectionRegex = new RegExp(`(?:^|\\n)(#{1,3}\\s*${section}[^\\n]*)([\\s\\S]*?)(?=\\n#{1,3}\\s|$)`, "i");
        const match = content.match(sectionRegex);
        if (match) {
          resultContent = match[1] + match[2];
        }
      }

      const isLarge = resultContent.length > 10000;

      return JSON.stringify({
        success: true,
        document: { id: doc.id, title: doc.title, category: doc.category },
        contentLength: resultContent.length,
        warning: isLarge ? "Large document - consider using search tools" : undefined,
        content: isLarge ? resultContent.substring(0, 10000) + "\n\n[TRUNCATED]" : resultContent,
      });
    } catch (error) {
      return JSON.stringify({
        success: false,
        error: `Failed to get document: ${error instanceof Error ? error.message : "Unknown error"}`,
      });
    }
  },
  {
    name: "getDocumentByName",
    description: "Get full document content by name. WARNING: Only use for small documents.",
    schema: z.object({
      documentName: z.string().describe("Document title or filename"),
      section: z.string().optional().describe("Specific section heading to extract"),
    }),
  }
);

/** All backend tools for the researcher */
export const researcherTools = [
  webSearch,
  listDocuments,
  searchDocuments,
  searchDocumentsByText,
  getDocumentLines,
  getDocumentByName,
];

// ============================================================================
// SYSTEM PROMPT
// ============================================================================

const RESEARCHER_SYSTEM_PROMPT = `You are The Researcher - a specialized agent focused on deep knowledge gathering for online training projects.

## CRITICAL: IMMEDIATE ACTION REQUIRED

**When you receive control from the orchestrator, you MUST immediately use tools to conduct research.**

❌ DO NOT respond with text like "I'll research this..." or "Let me gather information..."
❌ DO NOT explain what you're going to do - just DO IT
✅ IMMEDIATELY call web_search or document search tools in your FIRST response

Example: If the orchestrator says "Research heavy vehicle regulations", your response should START with tool calls:
- Call web_search("heavy vehicle regulations Australia NHVR")
- Call listDocuments() to check for relevant uploaded docs

NOT: "I'll research heavy vehicle regulations..." (THIS IS WRONG)

## Your Role

You conduct thorough research to provide comprehensive information that will inform course structure and content. Your research covers:

1. **Industry Context** - Current state, trends, challenges in the target industry
2. **Key Topics** - Core concepts, skills, and knowledge areas to cover
3. **Regulations & Compliance** - Relevant laws, standards, certifications
4. **Learner Personas** - Insights about the target audience's needs and challenges
5. **Best Practices** - Industry-standard approaches and methodologies

## Your Tools

### Web Search
- **web_search** - Search the web via Perplexity AI for current information, trends, regulations

### Document RAG
- **listDocuments** - See available uploaded documents
- **searchDocuments** - Semantic/conceptual search across documents
- **searchDocumentsByText** - Exact term/phrase search
- **getDocumentLines** - Get specific line ranges from documents
- **getDocumentByName** - Get full document (use sparingly for small docs)

### Media Library (Frontend)
- **searchMicroverse** - Search existing media assets for reference materials
- **getMicroverseDetails** - Get details about specific media assets

## CRITICAL: Research Limits & Completion

**⚠️ MANDATORY LIMITS - DO NOT EXCEED:**

1. **Maximum 3 web_search calls TOTAL** - Plan your searches upfront. Each search should cover broad ground.
2. **Maximum 2 document search calls** - Only if internal docs exist and are relevant
3. **NO "let me search for more" loops** - Do NOT say "Now let me search for more information on X" and make additional searches. Resist the urge to be exhaustive.
4. **ONE research phase** - Gather what you need in ONE batch of tool calls, then STOP and synthesize

**🛑 ANTI-PATTERN - DO NOT DO THIS:**
❌ "Now let me search for more information on [related topic]..."
❌ "I should also research [additional subtopic]..."
❌ "Let me dig deeper into [tangential area]..."
❌ Making tool calls after you already have tool results

**✓ CORRECT PATTERN:**
1. Make 1-3 web searches in parallel covering the main topic
2. Optionally check internal documents (1-2 searches max)
3. IMMEDIATELY synthesize findings and output [DONE]

**WHEN YOU ARE DONE RESEARCHING (MANDATORY):**
- Output your research findings in the JSON format below
- Include the marker **[DONE]** at the END of your response
- This signals you are ready to hand control back to the orchestrator
- **If you have made 3+ searches, you MUST output [DONE] NOW - no more searching**

**Example completion:**
\`\`\`
Based on my research, here are the findings:

\`\`\`json
{...your research JSON...}
\`\`\`

I have gathered sufficient information on the key topics, regulations, and industry context.

[DONE]
\`\`\`

## Research Strategy

1. **ACT IMMEDIATELY** - Start tool calls in your FIRST response, don't explain first
2. **Start Broad** - Use web_search for industry overview and current context
3. **Check Internal Docs** - Use listDocuments to see what's already available
4. **Deep Dive Critical Topics** - Focus on the most important areas only
5. **Synthesize & Complete** - After gathering enough info, produce findings and mark [DONE]

**REMEMBER: Your first response must contain tool calls, not explanatory text.**

## Output Format

Structure your findings as a research brief:

\`\`\`json
{
  "industryContext": "Overview of the industry...",
  "keyTopics": [
    { "topic": "Topic Name", "summary": "...", "importance": "critical|important|supplementary" }
  ],
  "regulations": [
    { "name": "Regulation Name", "summary": "...", "relevance": "..." }
  ],
  "personaInsights": "What we know about the learners...",
  "bestPractices": ["Practice 1", "Practice 2"],
  "citations": [
    { "title": "Source Title", "url": "...", "summary": "Key point from this source" }
  ]
}
\`\`\`

## Guidelines

- **EFFICIENCY IS KEY** - Get what you need in minimal searches, don't over-research
- Be thorough but focused - research what's relevant to the project brief
- Always cite sources for factual claims
- Distinguish between critical, important, and supplementary topics
- Note any gaps in available information
- Prioritize recent/current information over dated sources
- Check both web sources and internal documents
- Keep context size manageable - summarize rather than dump raw content
- **Always end with [DONE] when your research is complete**

Remember: Quality research leads to impactful training. Be efficient and focused.`;

// ============================================================================
// RESEARCHER NODE FUNCTION
// ============================================================================

// Maximum research iterations before forced completion
const MAX_RESEARCH_ITERATIONS = 8;

/**
 * The Researcher agent node.
 * Conducts deep research using web search and document RAG.
 */
export async function researcherNode(
  state: OrchestratorState,
  config: RunnableConfig
): Promise<Partial<OrchestratorState>> {
  console.log("\n[researcher] ============ Researcher Agent ============");
  console.log("  Project brief available:", state.projectBrief ? "yes" : "no");
  
  // Track research iterations
  const currentIteration = (state.researchIterationCount || 0) + 1;
  console.log(`  Research iteration: ${currentIteration}/${MAX_RESEARCH_ITERATIONS}`);

  // Build context-aware system message
  let systemContent = RESEARCHER_SYSTEM_PROMPT;

  // Count previous web_search calls in the conversation
  const messages = state.messages || [];
  let webSearchCount = 0;
  let docSearchCount = 0;
  for (const msg of messages) {
    const msgType = (msg as any)._getType?.() || (msg as any).constructor?.name || "";
    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolName = (msg as any).name || "";
      if (toolName === "web_search") webSearchCount++;
      if (toolName === "searchDocuments" || toolName === "searchDocumentsByText") docSearchCount++;
    }
  }
  
  const totalSearches = webSearchCount + docSearchCount;

  // Add iteration context to system prompt
  systemContent += `\n\n## Current Research Status
- **Iteration**: ${currentIteration} of ${MAX_RESEARCH_ITERATIONS} maximum
- **Web searches used**: ${webSearchCount}/3 maximum
- **Document searches used**: ${docSearchCount}/2 maximum  
- **Total searches**: ${totalSearches}`;

  if (totalSearches >= 3) {
    systemContent += `
- **⚠️ SEARCH LIMIT REACHED**: You have made ${totalSearches} searches. DO NOT make more tool calls. Output your findings and [DONE] NOW.`;
  }

  if (currentIteration >= MAX_RESEARCH_ITERATIONS - 2) {
    systemContent += `
- **WARNING**: You are approaching the iteration limit. Synthesize your findings NOW and output [DONE].`;
  }

  if (currentIteration >= MAX_RESEARCH_ITERATIONS) {
    systemContent += `
- **CRITICAL**: This is your FINAL iteration. You MUST output your research findings and [DONE] now. No more tool calls allowed.`;
  }

  // Include project brief if available
  if (state.projectBrief) {
    const condensedBrief = getCondensedBrief(state.projectBrief);
    systemContent += `\n\n## Project Brief to Research\n\n${condensedBrief}`;
  }

  // Include existing research if partial
  if (state.researchFindings) {
    systemContent += `\n\n## Existing Research (to extend)\n
Topics already researched: ${state.researchFindings.keyTopics.length}
Citations gathered: ${state.researchFindings.citations.length}

Continue researching to fill any gaps or go deeper on key topics.`;
  }

  const systemMessage = new SystemMessage({ content: systemContent });

  // Get frontend media tools from CopilotKit state
  const frontendActions = state.copilotkit?.actions ?? [];
  const mediaTools = frontendActions.filter((action: { name: string }) =>
    ["searchMicroverse", "getMicroverseDetails"].includes(action.name)
  );

  // Combine backend research tools with frontend media tools
  const allResearcherTools = [...researcherTools, ...mediaTools];

  // Bind all tools
  const modelWithTools = researcherModel.bindTools(allResearcherTools);

  // Filter messages for this agent's context - filter orphans first, then slice
  // Context management pipeline for researcher (heavy tool usage):
  // 1. Clear old tool results to prevent context bloat from large search/document outputs
  // 2. Slice to recent messages
  // 3. Filter orphaned tool results
  const clearedMessages = clearOldToolResults(state.messages || [], {
    keepCount: 5, // Keep last 5 tool results
    logPrefix: "[researcher]",
  });
  const slicedMessages = clearedMessages.slice(-MESSAGE_LIMITS.subAgent);
  const recentMessages = filterOrphanedToolResults(slicedMessages, "[researcher]");

  console.log("  Invoking researcher model with", researcherTools.length, "tools...");

  let response = await modelWithTools.invoke(
    [systemMessage, ...recentMessages],
    config
  );

  console.log("  Researcher response received");

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
1. Use the web_search tool to research the project topic
2. OR use document search tools to find relevant internal content
3. Provide a brief status update to the user

The user is waiting for research results.`,
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

  // Check if research is complete (contains [DONE] marker)
  const responseText = typeof aiResponse.content === "string"
    ? aiResponse.content
    : Array.isArray(aiResponse.content)
    ? aiResponse.content
        .filter((b): b is { type: "text"; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text")
        .map((b) => b.text)
        .join("\n")
    : "";
  
  const isResearchComplete = responseText.toLowerCase().includes("[done]");
  
  if (isResearchComplete) {
    console.log("  [researcher] Research complete - [DONE] marker found");
  }
  
  // If at max iterations and still making tool calls, force completion
  const forceComplete = currentIteration >= MAX_RESEARCH_ITERATIONS && aiResponse.tool_calls?.length;
  if (forceComplete) {
    console.log("  [researcher] MAX ITERATIONS reached - forcing completion");
  }

  return {
    messages: [response],
    currentAgent: "researcher",
    agentHistory: ["researcher"],
    // Clear routing decision when this agent starts - prevents stale routing
    routingDecision: null,
    // Update iteration count (reset to 0 if complete, otherwise increment)
    researchIterationCount: isResearchComplete ? 0 : currentIteration,
  };
}

/**
 * Parses a researcher's text response to extract research findings.
 */
export function parseResearchFindings(content: string): ResearchBrief | null {
  try {
    // Look for JSON block
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      return validateResearchBrief(parsed);
    }
    return null;
  } catch (error) {
    console.error("[researcher] Failed to parse research findings:", error);
    return null;
  }
}

function validateResearchBrief(input: Partial<ResearchBrief>): ResearchBrief {
  return {
    industryContext: input.industryContext || "",
    keyTopics: input.keyTopics || [],
    regulations: input.regulations || [],
    personaInsights: input.personaInsights || "",
    bestPractices: input.bestPractices || [],
    citations: input.citations || [],
    rawNotes: input.rawNotes,
  };
}

export default researcherNode;

