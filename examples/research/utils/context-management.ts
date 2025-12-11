/**
 * Context Management Utilities
 *
 * Centralized context management functions for the multi-agent orchestrator.
 * Provides message filtering, trimming, summarization, and tool result clearing.
 *
 * Based on LangChain middleware patterns adapted for StateGraph usage:
 * - trimMessages: Token-aware message trimming
 * - filterOrphanedToolResults: Anthropic API compatibility
 * - summarizeIfNeeded: Automatic conversation summarization
 * - clearOldToolResults: Tool output context management
 */

import { trimMessages as langchainTrimMessages } from "@langchain/core/messages";
import {
  AIMessage,
  SystemMessage,
  HumanMessage,
  BaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default token limits for different agent types */
export const TOKEN_LIMITS = {
  orchestrator: 40000,  // Reduced to prevent state explosion during FileSystemPersistence
  subAgent: 20000,      // Reduced to keep sub-agent context tight
  summary: 2000,
} as const;

/** Default message counts for fallback when token counting unavailable */
export const MESSAGE_LIMITS = {
  orchestrator: 20,     // Reduced aggressively - state persistence limit is 40 messages
  subAgent: 8,          // Keep sub-agent context minimal
} as const;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Gets the message type string from a BaseMessage.
 */
export function getMessageType(msg: BaseMessage): string {
  return (msg as any)._getType?.() || (msg as any).constructor?.name || "";
}

/**
 * Checks if an AI message has non-empty text content.
 * Messages with only thinking blocks (no text) cause Anthropic API errors.
 */
export function hasNonEmptyTextContent(msg: AIMessage): boolean {
  const content = msg.content;

  // String content - check if non-empty
  if (typeof content === "string") {
    return content.trim().length > 0;
  }

  // Array content - look for non-empty text blocks
  if (Array.isArray(content)) {
    for (const block of content as any[]) {
      if (typeof block === "string" && block.trim().length > 0) {
        return true;
      }
      if (typeof block === "object" && block !== null) {
        // Check for text block with content
        if ("type" in block && block.type === "text" && "text" in block) {
          const text = block.text;
          if (typeof text === "string" && text.trim().length > 0) {
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
 * Checks if an AI response has usable content (text or tool calls).
 * Returns false for empty responses or responses with only thinking blocks.
 */
export function hasUsableResponse(response: AIMessage): boolean {
  // Check for tool calls
  if (response.tool_calls && response.tool_calls.length > 0) {
    return true;
  }

  return hasNonEmptyTextContent(response);
}

/**
 * Strips thinking blocks from messages.
 * Required when an agent has thinking DISABLED but receives messages from
 * agents that have thinking ENABLED.
 */
export function stripThinkingBlocks(messages: BaseMessage[]): BaseMessage[] {
  return messages.map((msg) => {
    const msgType = getMessageType(msg);

    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = msg as AIMessage;
      const content = aiMsg.content;

      if (Array.isArray(content)) {
        // Filter out thinking blocks from content array
        const filteredContent = (content as any[]).filter((block) => {
          if (typeof block === "object" && block !== null && "type" in block) {
            return block.type !== "thinking" && block.type !== "redacted_thinking";
          }
          return true;
        });

        // If content changed, create a new message with filtered content
        // Don't copy additional_kwargs/response_metadata - they may contain
        // provider-specific data that shouldn't be re-sent
        if (filteredContent.length !== content.length) {
          return new AIMessage({
            content: filteredContent,
            tool_calls: aiMsg.tool_calls,
            id: aiMsg.id,
          });
        }
      }
    }

    return msg;
  });
}

/**
 * Strips tool_use blocks from message content.
 * CRITICAL: Anthropic API errors occur when both tool_calls property AND tool_use
 * blocks exist in content with the same IDs. This function removes tool_use from
 * content when we're relying on the tool_calls property instead.
 *
 * @param content - Message content (string or array)
 * @param toolCallIds - Optional set of tool_call IDs to specifically remove
 * @returns Content with tool_use blocks removed
 */
function stripToolUseFromContent(
  content: string | any[],
  toolCallIds?: Set<string>
): string | any[] {
  // String content - no tool_use to strip
  if (typeof content === "string") {
    return content;
  }

  // Array content - filter out tool_use blocks
  if (Array.isArray(content)) {
    const filtered = content.filter((block) => {
      if (typeof block === "object" && block !== null && "type" in block) {
        if (block.type === "tool_use") {
          // If toolCallIds provided, only remove matching ones
          if (toolCallIds && block.id) {
            return !toolCallIds.has(block.id);
          }
          // Otherwise remove all tool_use blocks
          return false;
        }
      }
      return true;
    });

    // If we filtered everything out, return empty string
    if (filtered.length === 0) {
      return "";
    }

    // If only one text block remains, simplify to string
    if (filtered.length === 1) {
      const block = filtered[0];
      if (typeof block === "string") {
        return block;
      }
      if (typeof block === "object" && block.type === "text" && block.text) {
        return block.text;
      }
    }

    return filtered;
  }

  return content;
}

// ============================================================================
// FILTER ORPHANED TOOL RESULTS
// ============================================================================

/**
 * Filters out orphaned messages to prevent Anthropic API errors:
 * - tool_result without matching tool_use
 * - AI messages with ONLY unresolved tool_calls and no text content
 *
 * IMPORTANT: When an AI message has SOME resolved and SOME unresolved tool_calls,
 * we create a NEW AI message with only the resolved ones to preserve the tool_use
 * for existing tool_results.
 *
 * @param messages - Array of messages to filter
 * @param logPrefix - Optional prefix for log messages (e.g., "[orchestrator]")
 * @returns Filtered array of messages
 */
export function filterOrphanedToolResults(
  messages: BaseMessage[],
  logPrefix: string = ""
): BaseMessage[] {
  const filtered: BaseMessage[] = [];
  const prefix = logPrefix ? `${logPrefix} ` : "";

  // First pass: collect all tool_result IDs in the message history
  const toolResultIds = new Set<string>();
  for (const msg of messages) {
    const msgType = getMessageType(msg);
    if (msgType === "tool" || msgType === "ToolMessage") {
      toolResultIds.add((msg as ToolMessage).tool_call_id);
    }
  }

  // Track tool_call IDs we've already processed to deduplicate
  // CRITICAL: CopilotKit with emitToolCalls:true creates duplicate AIMessages
  // - One from Claude's original response (with content + tool_calls)
  // - One from ActionExecutionMessage conversion (with empty content + tool_calls)
  // Both have the SAME tool_call IDs, causing Anthropic API errors
  const processedToolCallIds = new Set<string>();

  // Track tool_call_ids that already have a ToolMessage result
  // Anthropic API requires exactly ONE tool_result per tool_use
  const processedToolResultIds = new Set<string>();

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const msgType = getMessageType(msg);

    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = msg as AIMessage;

      // Check for tool_calls that need filtering
      if (aiMsg.tool_calls?.length) {
        // Check if ALL tool_calls in this message have already been processed
        // If so, this is a duplicate AIMessage from CopilotKit - skip it entirely
        const allToolCallIds = aiMsg.tool_calls.map(tc => tc.id).filter(Boolean);
        const allAlreadyProcessed = allToolCallIds.length > 0 && 
          allToolCallIds.every(id => processedToolCallIds.has(id!));
        
        if (allAlreadyProcessed) {
          console.log(
            `  ${prefix}[FILTER] Skipping duplicate AIMessage - tool_calls already processed: ${aiMsg.tool_calls.map(tc => tc.name).join(", ")}`
          );
          continue;
        }

        const resolvedToolCalls = aiMsg.tool_calls.filter(
          (tc) => tc.id && toolResultIds.has(tc.id)
        );
        const unresolvedToolCalls = aiMsg.tool_calls.filter(
          (tc) => tc.id && !toolResultIds.has(tc.id)
        );

        if (unresolvedToolCalls.length > 0) {
          console.log(
            `  ${prefix}[FILTER] Found ${unresolvedToolCalls.length} unresolved tool_calls: ${unresolvedToolCalls.map((tc) => tc.name).join(", ")}`
          );
        }

        // If we have resolved tool_calls, create a sanitized AI message
        // CRITICAL: Always create a new message to avoid duplicate tool_use IDs.
        // Claude returns AI messages with tool_use in BOTH content AND tool_calls.
        // If we pass the original, LangChain serializes both - causing Anthropic API errors.
        if (resolvedToolCalls.length > 0) {
          // Mark these tool_call IDs as processed to prevent duplicates
          resolvedToolCalls.forEach(tc => {
            if (tc.id) processedToolCallIds.add(tc.id);
          });

          // Get all tool_call IDs (resolved + unresolved) to strip from content
          const allToolCallIdSet = new Set<string>();
          aiMsg.tool_calls?.forEach(tc => { if (tc.id) allToolCallIdSet.add(tc.id); });

          // CRITICAL: Strip ALL tool_use blocks from content that match any tool_call
          // This prevents duplicate tool_use IDs when LangChain serializes tool_calls
          const strippedContent = stripToolUseFromContent(aiMsg.content, allToolCallIdSet);

          // Always create new AI message with stripped content + tool_calls
          const newAiMsg = new AIMessage({
            content: strippedContent,
            tool_calls: resolvedToolCalls,
            id: aiMsg.id,
            name: aiMsg.name,
          });
          filtered.push(newAiMsg);
          
          if (unresolvedToolCalls.length > 0) {
            console.log(
              `  ${prefix}[FILTER] Sanitized AI message: ${resolvedToolCalls.length} resolved, ${unresolvedToolCalls.length} unresolved tool_calls removed`
            );
          } else {
            console.log(
              `  ${prefix}[FILTER] Sanitized AI message with ${resolvedToolCalls.length} tool_calls (stripped tool_use from content)`
            );
          }
          continue;
        }

        // No resolved tool_calls - if has text content, strip the unresolved tool_calls
        if (hasNonEmptyTextContent(aiMsg)) {
          // Get IDs to strip from content
          const unresolvedIdSet = new Set<string>();
          unresolvedToolCalls.forEach(tc => { if (tc.id) unresolvedIdSet.add(tc.id); });
          
          // Strip tool_use blocks from content
          const strippedContent = stripToolUseFromContent(aiMsg.content, unresolvedIdSet);
          
          // Create new message without tool_calls to prevent API errors
          const newAiMsg = new AIMessage({
            content: strippedContent,
            id: aiMsg.id,
            name: aiMsg.name,
            // Explicitly omit tool_calls since they're all unresolved
          });
          filtered.push(newAiMsg);
          console.log(
            `  ${prefix}[FILTER] Stripped ${unresolvedToolCalls.length} unresolved tool_calls from AI message with content: ${unresolvedToolCalls.map((tc) => tc.name).join(", ")}`
          );
          continue;
        }
        
        // No resolved tool_calls AND no text content - remove entirely
        console.log(
          `  ${prefix}[FILTER] Removing AI message with no content and no resolved tool_calls`
        );
        continue;
      }

      // CRITICAL: ALWAYS strip tool_use blocks from content when there are no tool_calls.
      // These are orphaned tool_use blocks that will cause "tool_use ids must be unique" errors
      // if they happen to share IDs with tool_use in other messages.
      // When there are no tool_calls, tool_use in content serves no purpose anyway.
      if (Array.isArray(aiMsg.content)) {
        const hasToolUse = (aiMsg.content as any[]).some(
          block => typeof block === "object" && block !== null && block.type === "tool_use"
        );
        if (hasToolUse) {
          const strippedContent = stripToolUseFromContent(aiMsg.content);
          
          // After stripping, check if there's any content left
          const hasContentAfterStrip = typeof strippedContent === "string" 
            ? strippedContent.trim().length > 0
            : Array.isArray(strippedContent) && strippedContent.length > 0;
          
          if (hasContentAfterStrip) {
            const newAiMsg = new AIMessage({
              content: strippedContent,
              id: aiMsg.id,
              name: aiMsg.name,
            });
            filtered.push(newAiMsg);
            console.log(`  ${prefix}[FILTER] Stripped orphaned tool_use blocks from AI message (no tool_calls)`);
            continue;
          } else {
            // Content was only tool_use blocks with no text - remove entirely
            console.log(`  ${prefix}[FILTER] Removing AI message with only orphaned tool_use blocks`);
            continue;
          }
        }
      }
      
      // Filter AI messages with empty text content (only thinking blocks)
      if (!hasNonEmptyTextContent(aiMsg)) {
        console.log(
          `  ${prefix}[FILTER] Removing AI message with empty content`
        );
        continue;
      }
      
      filtered.push(msg);
      continue;
    }

    // Filter orphaned tool_result messages (results without matching tool_use)
    // CRITICAL: Anthropic requires tool_result to match tool_use in IMMEDIATELY preceding assistant message
    // Multiple consecutive ToolMessages can follow one AIMessage (valid), but the AIMessage must be adjacent
    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      const toolCallId = toolMsg.tool_call_id;

      // Check for duplicate tool results (same tool_call_id)
      // Anthropic API requires exactly ONE tool_result per tool_use
      if (processedToolResultIds.has(toolCallId)) {
        console.log(`  ${prefix}[FILTER] Removing duplicate tool result for: ${toolCallId}`);
        continue;
      }

      let hasMatchingToolUse = false;
      
      // Find the most recent non-tool message (must be an AI message with matching tool_use)
      // We allow consecutive ToolMessages because they all reference the same preceding AIMessage
      for (let j = filtered.length - 1; j >= 0; j--) {
        const prevMsg = filtered[j];
        const prevType = getMessageType(prevMsg);

        // Skip other tool messages (consecutive tool results from same AI message are valid)
        if (prevType === "tool" || prevType === "ToolMessage") {
          continue;
        }

        // Found a non-tool message - must be an AI message with matching tool_use
        if (prevType === "ai" || prevType === "AIMessage" || prevType === "AIMessageChunk") {
          const aiMsg = prevMsg as AIMessage;
          if (aiMsg.tool_calls?.some((tc) => tc.id === toolCallId)) {
            hasMatchingToolUse = true;
          }
        }
        // Only check the immediately preceding non-tool message (strict adjacency)
        break;
      }

      if (!hasMatchingToolUse) {
        console.log(`  ${prefix}[FILTER] Removing orphaned tool result (no adjacent tool_use): ${toolCallId}`);
        continue;
      }

      // Mark this tool_call_id as having a result
      processedToolResultIds.add(toolCallId);
    }

    filtered.push(msg);
  }

  return filtered;
}

// ============================================================================
// REPAIR DANGLING TOOL CALLS (Deep Agents Pattern)
// ============================================================================

/**
 * Repairs message history when tool calls have no corresponding results.
 * Based on LangGraph Deep Agents "Dangling Tool Call Repair" pattern.
 *
 * The problem:
 * - Agent requests tool call via AIMessage with tool_calls
 * - Tool call is interrupted (user cancels, error, trimming, etc.)
 * - AIMessage has tool_use but no corresponding ToolMessage result
 * - This creates an invalid message sequence for Claude/Anthropic
 *
 * The solution:
 * - Detects AIMessages with tool_calls that have no results
 * - Creates synthetic ToolMessage responses indicating the call was cancelled
 * - Repairs the message history before agent execution
 *
 * @param messages - Array of messages to repair
 * @param logPrefix - Optional prefix for log messages
 * @returns Repaired array of messages with synthetic tool results
 */
export function repairDanglingToolCalls(
  messages: BaseMessage[],
  logPrefix: string = ""
): BaseMessage[] {
  const prefix = logPrefix ? `${logPrefix} ` : "";

  // First pass: collect all tool_result IDs in the message history
  const toolResultIds = new Set<string>();
  for (const msg of messages) {
    const msgType = getMessageType(msg);
    if (msgType === "tool" || msgType === "ToolMessage") {
      toolResultIds.add((msg as ToolMessage).tool_call_id);
    }
  }

  // Second pass: find AIMessages with unresolved tool_calls and insert synthetic results
  const repaired: BaseMessage[] = [];
  let repairedCount = 0;

  for (const msg of messages) {
    repaired.push(msg);

    const msgType = getMessageType(msg);
    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = msg as AIMessage;

      // Collect all tool calls from BOTH sources:
      // 1. The tool_calls property (standard LangChain format)
      // 2. The content array (may contain tool_use blocks from Claude)
      const allToolCalls: Array<{ id: string; name: string }> = [];

      // Check tool_calls property
      if (aiMsg.tool_calls?.length) {
        for (const tc of aiMsg.tool_calls) {
          if (tc.id) {
            allToolCalls.push({ id: tc.id, name: tc.name });
          }
        }
      }

      // Also check content array for tool_use blocks (Claude's native format)
      // These might exist even if tool_calls is empty
      if (Array.isArray(aiMsg.content)) {
        for (const block of aiMsg.content as any[]) {
          if (typeof block === "object" && block !== null && block.type === "tool_use" && block.id) {
            // Only add if not already in allToolCalls
            if (!allToolCalls.some(tc => tc.id === block.id)) {
              allToolCalls.push({ id: block.id, name: block.name || "unknown_tool" });
            }
          }
        }
      }

      if (allToolCalls.length > 0) {
        // Find tool_calls without corresponding results
        const danglingToolCalls = allToolCalls.filter(
          (tc) => !toolResultIds.has(tc.id)
        );

        // Create synthetic ToolMessage for each dangling call
        for (const toolCall of danglingToolCalls) {
          const syntheticResult = new ToolMessage({
            content: `[Tool call cancelled or interrupted - no result available]`,
            tool_call_id: toolCall.id,
            name: toolCall.name,
          });

          repaired.push(syntheticResult);
          toolResultIds.add(toolCall.id); // Mark as resolved
          repairedCount++;
        }
      }
    }
  }

  if (repairedCount > 0) {
    console.log(
      `  ${prefix}[REPAIR] Created ${repairedCount} synthetic tool results for dangling tool calls`
    );
  }

  return repaired;
}

// ============================================================================
// TRIM MESSAGES (LangChain Integration)
// ============================================================================

export interface TrimMessagesOptions {
  /** Maximum tokens to keep (default: TOKEN_LIMITS.orchestrator) */
  maxTokens?: number;
  /** Model to use for token counting (optional) */
  model?: BaseChatModel;
  /** Fallback message count if token counting unavailable */
  fallbackMessageCount?: number;
  /** Strategy for trimming: "last" keeps most recent */
  strategy?: "last" | "first";
  /** Message type to start the trimmed window on */
  startOn?: "human" | "ai" | "system";
  /** Message types that can end the trimmed window */
  endOn?: ("human" | "ai" | "tool")[];
  /** Log prefix for debugging */
  logPrefix?: string;
  /** Original user request to preserve (from activeTask) */
  originalRequest?: string;
  /** Keywords that indicate task-critical messages to preserve */
  preserveKeywords?: string[];
}

/**
 * Trims message history using LangChain's trimMessages utility.
 * Falls back to message count if token counting is unavailable.
 *
 * Preserves:
 * - System messages (always kept at start)
 * - Tool use/result pairs (never breaks them apart)
 * - Task-critical messages (containing original user request or keywords)
 * - Recent context based on token or message count
 *
 * @param messages - Messages to trim
 * @param options - Trimming options
 * @returns Trimmed messages
 */
export async function trimMessages(
  messages: BaseMessage[],
  options: TrimMessagesOptions = {}
): Promise<BaseMessage[]> {
  const {
    maxTokens = TOKEN_LIMITS.orchestrator,
    model,
    fallbackMessageCount = MESSAGE_LIMITS.orchestrator,
    strategy = "last",
    startOn = "human",
    endOn = ["human", "tool"],
    logPrefix = "",
    originalRequest,
    preserveKeywords = [],
  } = options;

  const prefix = logPrefix ? `${logPrefix} ` : "";

  // Helper: check if a message is task-critical and should be preserved
  const isTaskCritical = (msg: BaseMessage): boolean => {
    const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
    
    // Check if this message contains the original user request
    if (originalRequest && content.includes(originalRequest.substring(0, 100))) {
      return true;
    }
    
    // Check for preserve keywords
    const contentLower = content.toLowerCase();
    for (const keyword of preserveKeywords) {
      if (contentLower.includes(keyword.toLowerCase())) {
        return true;
      }
    }
    
    return false;
  };

  // If we have a model, use token-based trimming
  if (model) {
    try {
      const trimmed = await langchainTrimMessages(messages, {
        maxTokens,
        strategy,
        startOn,
        endOn,
        tokenCounter: model,
      });

      console.log(`  ${prefix}[TRIM] Token-based: ${messages.length} -> ${trimmed.length} messages`);
      return trimmed;
    } catch (error) {
      console.warn(`  ${prefix}[TRIM] Token counting failed, falling back to message count:`, error);
    }
  }

  // Fallback: Message count-based trimming (original logic)
  if (messages.length <= fallbackMessageCount) {
    return messages;
  }

  // Always keep system messages
  const systemMessages = messages.filter((m) => {
    const msgType = getMessageType(m);
    return msgType === "system" || msgType === "SystemMessage";
  });

  const otherMessages = messages.filter((m) => {
    const msgType = getMessageType(m);
    return msgType !== "system" && msgType !== "SystemMessage";
  });

  // Identify task-critical messages that should be preserved
  const taskCriticalIndices = new Set<number>();
  otherMessages.forEach((msg, idx) => {
    if (isTaskCritical(msg)) {
      taskCriticalIndices.add(idx);
    }
  });

  if (taskCriticalIndices.size > 0) {
    console.log(`  ${prefix}[TRIM] Found ${taskCriticalIndices.size} task-critical messages to preserve`);
  }

  // Find a safe cut point that doesn't break tool_use/tool_result pairs
  let startIdx = Math.max(0, otherMessages.length - fallbackMessageCount);

  // Scan forward to find a safe starting point (human message is always safe)
  while (startIdx < otherMessages.length) {
    const msg = otherMessages[startIdx];
    const msgType = getMessageType(msg);

    // Safe to start at: human message, or AI message without pending tool calls
    if (msgType === "human" || msgType === "HumanMessage") {
      break;
    }

    // Tool messages need their corresponding AI tool_use message
    if (msgType === "tool" || msgType === "ToolMessage") {
      startIdx++;
      continue;
    }

    // AI messages are safe if they don't have tool calls that need results
    if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
      const aiMsg = msg as AIMessage;
      const hasToolCalls = aiMsg.tool_calls?.length > 0;
      if (!hasToolCalls) {
        break;
      }
      startIdx++;
      continue;
    }

    startIdx++;
  }

  // Safety: ensure we don't trim away ALL messages
  if (startIdx >= otherMessages.length && otherMessages.length > 0) {
    console.warn(`  ${prefix}[TRIM] WARNING: No valid start point found, keeping last ${fallbackMessageCount} messages`);
    startIdx = Math.max(0, otherMessages.length - fallbackMessageCount);
  }

  // Build result, preserving task-critical messages even if they're before startIdx
  const recentMessages: BaseMessage[] = [];
  const preservedFromOlderMessages: BaseMessage[] = [];

  for (let i = 0; i < otherMessages.length; i++) {
    const msg = otherMessages[i];
    
    if (i >= startIdx) {
      // Recent messages - always include
      recentMessages.push(msg);
    } else if (taskCriticalIndices.has(i)) {
      // Older but task-critical - preserve
      preservedFromOlderMessages.push(msg);
    }
  }

  // Combine: system messages + preserved older task-critical + recent
  // NOTE: We don't add separator messages because:
  // 1. SystemMessage would cause "System messages are only permitted as the first passed message" error
  // 2. HumanMessage would break turn-taking patterns and create orphaned messages
  // The preserved messages are self-explanatory, and agents have task context injection
  const result: BaseMessage[] = [...systemMessages, ...preservedFromOlderMessages, ...recentMessages];

  console.log(
    `  ${prefix}[TRIM] Messages: ${messages.length} -> ${result.length} (${preservedFromOlderMessages.length} task-critical preserved)`
  );

  // Final safety check: never return empty array if we had messages
  if (result.length === 0 && messages.length > 0) {
    console.warn(`  ${prefix}[TRIM] CRITICAL: Would return 0 messages, keeping original`);
    return messages.slice(-fallbackMessageCount);
  }

  return result;
}

// ============================================================================
// SUMMARIZATION
// ============================================================================

/** Model instance for generating summaries (lazy initialized) */
let summaryModel: ChatAnthropic | null = null;

function getSummaryModel(): ChatAnthropic {
  if (!summaryModel) {
    summaryModel = new ChatAnthropic({
      model: "claude-sonnet-4-5-20250929",
      maxTokens: TOKEN_LIMITS.summary,
      temperature: 0.3, // Lower temperature for consistent summaries
    });
  }
  return summaryModel;
}

export interface SummarizeOptions {
  /** Token threshold to trigger summarization */
  triggerTokens?: number;
  /** Number of recent messages to keep (not summarized) */
  keepMessages?: number;
  /** Custom model for summarization */
  model?: BaseChatModel;
  /** Custom summary prompt */
  summaryPrompt?: string;
  /** Prefix for the summary message */
  summaryPrefix?: string;
  /** Log prefix for debugging */
  logPrefix?: string;
}

const DEFAULT_SUMMARY_PROMPT = `Summarize the following conversation concisely, preserving:
1. Key decisions made
2. Important information exchanged
3. Current task or goal state
4. Any pending actions or questions

Be concise but complete. Focus on information the AI will need to continue the conversation effectively.

Conversation to summarize:`;

/**
 * Summarizes older messages when token/message limits are reached.
 * Preserves recent messages and replaces older ones with a summary.
 *
 * Based on LangChain's summarizationMiddleware pattern.
 *
 * @param messages - Messages to potentially summarize
 * @param options - Summarization options
 * @returns Messages with older ones summarized if threshold reached
 */
export async function summarizeIfNeeded(
  messages: BaseMessage[],
  options: SummarizeOptions = {}
): Promise<BaseMessage[]> {
  const {
    triggerTokens = 100000,
    keepMessages = 20,
    model = getSummaryModel(),
    summaryPrompt = DEFAULT_SUMMARY_PROMPT,
    summaryPrefix = "Previous conversation summary:",
    logPrefix = "",
  } = options;

  const prefix = logPrefix ? `${logPrefix} ` : "";

  // Quick check: if we have fewer messages than keepMessages, no summarization needed
  if (messages.length <= keepMessages) {
    return messages;
  }

  // Separate system messages (always keep at start)
  const systemMessages = messages.filter((m) => {
    const msgType = getMessageType(m);
    return msgType === "system" || msgType === "SystemMessage";
  });

  const conversationMessages = messages.filter((m) => {
    const msgType = getMessageType(m);
    return msgType !== "system" && msgType !== "SystemMessage";
  });

  // If conversation portion is small enough, no summarization needed
  if (conversationMessages.length <= keepMessages) {
    return messages;
  }

  // Estimate token count (rough: ~4 chars per token)
  const estimatedTokens = messages.reduce((sum, m) => {
    const content = typeof m.content === "string" 
      ? m.content 
      : JSON.stringify(m.content);
    return sum + Math.ceil(content.length / 4);
  }, 0);

  if (estimatedTokens < triggerTokens) {
    return messages;
  }

  console.log(`  ${prefix}[SUMMARIZE] Triggering summarization (est. ${estimatedTokens} tokens, ${messages.length} messages)`);

  // Split into messages to summarize vs keep
  const toSummarize = conversationMessages.slice(0, -keepMessages);
  const toKeep = conversationMessages.slice(-keepMessages);

  // Format messages for summarization
  const formattedMessages = toSummarize.map((m) => {
    const msgType = getMessageType(m);
    const role = msgType.includes("human") || msgType.includes("Human") 
      ? "User" 
      : msgType.includes("ai") || msgType.includes("AI") 
        ? "Assistant" 
        : "System";
    const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    return `${role}: ${content.substring(0, 500)}${content.length > 500 ? "..." : ""}`;
  }).join("\n\n");

  try {
    // Generate summary
    const summaryResponse = await model.invoke([
      new SystemMessage(summaryPrompt),
      new HumanMessage(formattedMessages),
    ]);

    const summaryText = typeof summaryResponse.content === "string"
      ? summaryResponse.content
      : JSON.stringify(summaryResponse.content);

    console.log(`  ${prefix}[SUMMARIZE] Created summary (${summaryText.length} chars) replacing ${toSummarize.length} messages`);

    // Create summary as a system message
    const summaryMessage = new SystemMessage({
      content: `${summaryPrefix}\n\n${summaryText}`,
    });

    // Return: system messages + summary + kept messages
    return [...systemMessages, summaryMessage, ...toKeep];
  } catch (error) {
    console.error(`  ${prefix}[SUMMARIZE] Failed to generate summary:`, error);
    // On failure, fall back to simple trimming
    return [...systemMessages, ...toKeep];
  }
}

// ============================================================================
// COMPRESS TOOL RESULTS (Reduce Token Usage)
// ============================================================================

/**
 * Tool-specific compression rules that transform verbose tool results into concise summaries.
 * Each function receives the raw tool result content and returns a compressed version.
 */
export type ToolCompressionRule = (content: string) => string;

export interface CompressToolResultsOptions {
  /** Number of most recent tool results to keep full (not compressed) */
  keepFullResultsCount?: number;
  /** Custom compression rules by tool name */
  compressionRules?: Record<string, ToolCompressionRule>;
  /** Maximum length for tool results before compression (chars) */
  maxResultLength?: number;
  /** Log prefix for debugging */
  logPrefix?: string;
}

/**
 * Default compression rules for common verbose tools.
 * These extract key information while dramatically reducing token count.
 */
const DEFAULT_COMPRESSION_RULES: Record<string, ToolCompressionRule> = {
  // Template tools - extract just the count and names
  getAvailableTemplates: (content: string) => {
    try {
      const data = JSON.parse(content);
      const templates = data.templates || [];
      const names = templates.slice(0, 5).map((t: any) => t.name).join(', ');
      const more = templates.length > 5 ? ` (+${templates.length - 5} more)` : '';
      return `[${templates.length} templates: ${names}${more}]`;
    } catch {
      return `[Templates data - ${content.length} chars]`;
    }
  },
  
  getNodeTemplateFields: (content: string) => {
    try {
      const data = JSON.parse(content);
      const fields = data.fields || [];
      const fieldNames = fields.map((f: any) => f.label || f.name).join(', ');
      return `[Template "${data.templateName}": ${fields.length} fields - ${fieldNames}]`;
    } catch {
      return `[Template fields - ${content.length} chars]`;
    }
  },
  
  // Image generation - keep just the file ID and title
  generateAIImage: (content: string) => {
    try {
      const data = JSON.parse(content);
      if (data.success && data.file) {
        return `[Image generated: id="${data.file.id}", title="${data.file.title}"]`;
      }
      return `[Image generation result - ${content.length} chars]`;
    } catch {
      return `[Image result - ${content.length} chars]`;
    }
  },
  
  // Node operations - keep key identifiers
  getNodeDetails: (content: string) => {
    try {
      const data = JSON.parse(content);
      return `[Node "${data.title || data.name}" (id: ${data.id}, type: ${data.nodeType})]`;
    } catch {
      return `[Node details - ${content.length} chars]`;
    }
  },
  
  getNodeChildren: (content: string) => {
    try {
      const data = JSON.parse(content);
      const children = data.children || data.nodes || [];
      const names = children.slice(0, 3).map((c: any) => c.title || c.name).join(', ');
      const more = children.length > 3 ? ` (+${children.length - 3} more)` : '';
      return `[${children.length} children: ${names}${more}]`;
    } catch {
      return `[Children data - ${content.length} chars]`;
    }
  },
  
  getNodesByLevel: (content: string) => {
    try {
      const data = JSON.parse(content);
      const nodes = data.nodes || [];
      const names = nodes.slice(0, 3).map((n: any) => n.title || n.name).join(', ');
      const more = nodes.length > 3 ? ` (+${nodes.length - 3} more)` : '';
      return `[${nodes.length} nodes at level: ${names}${more}]`;
    } catch {
      return `[Level nodes - ${content.length} chars]`;
    }
  },
  
  // Project/hierarchy info - summarize key fields
  getProjectHierarchyInfo: (content: string) => {
    try {
      const data = JSON.parse(content);
      const levels = data.hierarchyLevels || data.levels || [];
      const levelNames = levels.map((l: any) => l.name).join(' > ');
      return `[Hierarchy: ${levels.length} levels - ${levelNames}]`;
    } catch {
      return `[Hierarchy info - ${content.length} chars]`;
    }
  },
  
  getCurrentProject: (content: string) => {
    try {
      const data = JSON.parse(content);
      return `[Project: "${data.name}" (id: ${data.id})]`;
    } catch {
      return `[Project info - ${content.length} chars]`;
    }
  },
  
  // Search results - summarize
  searchMicroverse: (content: string) => {
    try {
      const data = JSON.parse(content);
      const results = data.results || data.files || [];
      return `[${results.length} media files found]`;
    } catch {
      return `[Search results - ${content.length} chars]`;
    }
  },
  
  // Node creation - keep the created node ID
  createNode: (content: string) => {
    try {
      const data = JSON.parse(content);
      if (data.success || data.nodeId || data.id) {
        const nodeId = data.nodeId || data.id || data.node?.id;
        const title = data.title || data.node?.title || 'untitled';
        return `[Node created: "${title}" (id: ${nodeId})]`;
      }
      return `[Create node result - ${content.length} chars]`;
    } catch {
      return `[Create result - ${content.length} chars]`;
    }
  },
  
  // Edit mode - simple status
  requestEditMode: (content: string) => {
    if (content.toLowerCase().includes('success')) {
      return '[Edit mode acquired]';
    }
    return `[Edit mode: ${content.substring(0, 100)}]`;
  },
  
  releaseEditMode: (content: string) => {
    return '[Edit mode released]';
  },
};

/**
 * Compresses older tool results to reduce token usage.
 * Keeps the most recent tool results full, compresses older ones using tool-specific rules.
 * 
 * This dramatically reduces context size for conversations with many tool calls,
 * particularly for verbose tools like getAvailableTemplates, generateAIImage, etc.
 * 
 * @param messages - Messages to process
 * @param options - Compression options
 * @returns Messages with older tool results compressed
 */
export function compressToolResults(
  messages: BaseMessage[],
  options: CompressToolResultsOptions = {}
): BaseMessage[] {
  const {
    keepFullResultsCount = 2,
    compressionRules = {},
    maxResultLength = 1000,
    logPrefix = "",
  } = options;

  const prefix = logPrefix ? `${logPrefix} ` : "";
  
  // Merge custom rules with defaults
  const allRules: Record<string, ToolCompressionRule> = {
    ...DEFAULT_COMPRESSION_RULES,
    ...compressionRules,
  };

  // Find all tool result messages (from newest to oldest)
  const toolResultIndices: { index: number; toolName: string; length: number }[] = [];

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const msgType = getMessageType(msg);

    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      const contentLength = typeof toolMsg.content === 'string' 
        ? toolMsg.content.length 
        : JSON.stringify(toolMsg.content).length;
      toolResultIndices.push({ 
        index: i, 
        toolName: toolMsg.name || 'unknown',
        length: contentLength,
      });
    }
  }

  // If we have fewer tool results than keepFullResultsCount, nothing to compress
  if (toolResultIndices.length <= keepFullResultsCount) {
    return messages;
  }

  // Identify which tool results to compress (older ones)
  const toCompress = new Set<number>();
  let keptCount = 0;

  for (const { index, length } of toolResultIndices) {
    if (keptCount < keepFullResultsCount) {
      keptCount++;
    } else if (length > maxResultLength) {
      // Only compress if over the length threshold
      toCompress.add(index);
    }
  }

  if (toCompress.size === 0) {
    return messages;
  }

  let compressedCount = 0;
  let savedChars = 0;

  // Build new message array with compressed tool results
  const result: BaseMessage[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (toCompress.has(i)) {
      const toolMsg = msg as ToolMessage;
      const toolName = toolMsg.name || 'unknown';
      const originalContent = typeof toolMsg.content === 'string' 
        ? toolMsg.content 
        : JSON.stringify(toolMsg.content);
      
      // Apply compression rule
      let compressedContent: string;
      if (allRules[toolName]) {
        compressedContent = allRules[toolName](originalContent);
      } else {
        // Default compression: truncate with length indicator
        compressedContent = originalContent.length > 200
          ? `${originalContent.substring(0, 200)}... [truncated, ${originalContent.length} chars total]`
          : originalContent;
      }
      
      savedChars += originalContent.length - compressedContent.length;
      compressedCount++;

      result.push(
        new ToolMessage({
          content: compressedContent,
          tool_call_id: toolMsg.tool_call_id,
          name: toolMsg.name,
        })
      );
      continue;
    }

    result.push(msg);
  }

  if (compressedCount > 0) {
    console.log(
      `  ${prefix}[COMPRESS] Compressed ${compressedCount} tool results, saved ~${Math.round(savedChars / 4)} tokens (${savedChars} chars)`
    );
  }

  return result;
}

// ============================================================================
// CLEAR OLD TOOL RESULTS
// ============================================================================

export interface ClearToolResultsOptions {
  /** Number of most recent tool results to keep */
  keepCount?: number;
  /** Tool names to exclude from clearing (always keep) */
  excludeTools?: string[];
  /** Whether to also clear the tool call arguments */
  clearToolInputs?: boolean;
  /** Placeholder text for cleared results */
  placeholder?: string;
  /** Log prefix for debugging */
  logPrefix?: string;
}

/**
 * Clears older tool results while preserving recent ones.
 * Based on LangChain's ClearToolUsesEdit pattern.
 *
 * This reduces context bloat from large tool outputs (web search, document content)
 * while maintaining enough recent context for the model.
 *
 * @param messages - Messages to process
 * @param options - Clearing options
 * @returns Messages with older tool results cleared
 */
export function clearOldToolResults(
  messages: BaseMessage[],
  options: ClearToolResultsOptions = {}
): BaseMessage[] {
  const {
    keepCount = 5,
    excludeTools = [],
    clearToolInputs = false,
    placeholder = "[Tool output cleared to save context]",
    logPrefix = "",
  } = options;

  const prefix = logPrefix ? `${logPrefix} ` : "";
  const excludeSet = new Set(excludeTools);

  // Find all tool result messages (from newest to oldest)
  const toolResultIndices: { index: number; toolName: string }[] = [];

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const msgType = getMessageType(msg);

    if (msgType === "tool" || msgType === "ToolMessage") {
      const toolMsg = msg as ToolMessage;
      toolResultIndices.push({ index: i, toolName: toolMsg.name });
    }
  }

  // If we have fewer tool results than keepCount, nothing to clear
  if (toolResultIndices.length <= keepCount) {
    return messages;
  }

  // Identify which tool results to clear (oldest ones, excluding protected tools)
  const toClear = new Set<number>();
  let keptCount = 0;

  for (const { index, toolName } of toolResultIndices) {
    // Always keep excluded tools
    if (excludeSet.has(toolName)) {
      continue;
    }

    if (keptCount < keepCount) {
      keptCount++;
    } else {
      toClear.add(index);
    }
  }

  if (toClear.size === 0) {
    return messages;
  }

  console.log(`  ${prefix}[CLEAR] Clearing ${toClear.size} old tool results, keeping ${keptCount}`);

  // Build new message array with cleared tool results
  const result: BaseMessage[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (toClear.has(i)) {
      // Replace tool result content with placeholder
      const toolMsg = msg as ToolMessage;
      result.push(
        new ToolMessage({
          content: placeholder,
          tool_call_id: toolMsg.tool_call_id,
          name: toolMsg.name,
        })
      );
      continue;
    }

    // Optionally clear tool call arguments in AI messages
    if (clearToolInputs) {
      const msgType = getMessageType(msg);
      if (msgType === "ai" || msgType === "AIMessage" || msgType === "AIMessageChunk") {
        const aiMsg = msg as AIMessage;
        if (aiMsg.tool_calls?.length) {
          // Check if any of this message's tool calls were cleared
          const hasCleared = aiMsg.tool_calls.some((tc) => {
            // Find the corresponding tool result
            const resultIdx = messages.findIndex(
              (m, idx) =>
                idx > i &&
                (getMessageType(m) === "tool" || getMessageType(m) === "ToolMessage") &&
                (m as ToolMessage).tool_call_id === tc.id
            );
            return resultIdx !== -1 && toClear.has(resultIdx);
          });

          if (hasCleared) {
            // Clear arguments for tool calls whose results were cleared
            const clearedToolCalls = aiMsg.tool_calls.map((tc) => {
              const resultIdx = messages.findIndex(
                (m, idx) =>
                  idx > i &&
                  (getMessageType(m) === "tool" || getMessageType(m) === "ToolMessage") &&
                  (m as ToolMessage).tool_call_id === tc.id
              );
              if (resultIdx !== -1 && toClear.has(resultIdx)) {
                return { ...tc, args: {} };
              }
              return tc;
            });

            // Don't copy content - with Anthropic, content contains tool_use blocks
            // that would duplicate with tool_calls when serialized.
            // Let LangChain rebuild content from tool_calls.
            result.push(
              new AIMessage({
                content: "",
                tool_calls: clearedToolCalls,
                id: aiMsg.id,
                name: aiMsg.name,
              })
            );
            continue;
          }
        }
      }
    }

    result.push(msg);
  }

  return result;
}

// ============================================================================
// COMBINED CONTEXT PROCESSING
// ============================================================================

export interface ProcessContextOptions {
  /** Maximum tokens (for trimming) */
  maxTokens?: number;
  /** Model for token counting */
  model?: BaseChatModel;
  /** Fallback message count */
  fallbackMessageCount?: number;
  /** Whether to apply summarization */
  enableSummarization?: boolean;
  /** Summarization trigger threshold */
  summarizeTriggerTokens?: number;
  /** Messages to keep when summarizing */
  summarizeKeepMessages?: number;
  /** Whether to clear old tool results */
  enableToolClearing?: boolean;
  /** Tool results to keep */
  toolKeepCount?: number;
  /** Tools to exclude from clearing */
  excludeTools?: string[];
  /** Whether to compress verbose tool results */
  enableToolCompression?: boolean;
  /** Number of recent tool results to keep full (not compressed) */
  compressionKeepCount?: number;
  /** Max length for tool results before compression */
  compressionMaxLength?: number;
  /** Log prefix */
  logPrefix?: string;
  /** Original user request to preserve (from activeTask) - messages containing this will be kept */
  originalRequest?: string;
  /** Keywords that indicate task-critical messages to preserve during trimming */
  preserveKeywords?: string[];
}

/**
 * Processes context by applying multiple strategies in sequence:
 * 1. Filter orphaned tool results (pre-trim cleanup)
 * 2. Repair dangling tool calls (create synthetic results for interrupted calls)
 * 3. Compress verbose tool results (if enabled) - REDUCES TOKEN USAGE
 * 4. Clear old tool results (if enabled)
 * 5. Summarize (if enabled)
 * 6. Trim to token/message limit
 * 7. Filter orphaned tool results again (post-trim cleanup)
 * 8. Repair dangling tool calls again (post-trim)
 *
 * Steps 7-8 are critical: trimMessages() fallback can create new orphans/danglers
 * when no valid start point is found and it falls back to keeping last N messages.
 *
 * This follows LangGraph's "Deep Agents Harness" patterns:
 * - Dangling tool call repair (AIMessage with tool_calls but no ToolMessage results)
 * - Orphan tool result filtering (ToolMessage without matching AIMessage tool_use)
 *
 * @param messages - Messages to process
 * @param options - Processing options
 * @returns Processed messages
 */
export async function processContext(
  messages: BaseMessage[],
  options: ProcessContextOptions = {}
): Promise<BaseMessage[]> {
  const {
    maxTokens = TOKEN_LIMITS.orchestrator,
    model,
    fallbackMessageCount = MESSAGE_LIMITS.orchestrator,
    enableSummarization = false,
    summarizeTriggerTokens = 100000,
    summarizeKeepMessages = 20,
    enableToolClearing = false,
    toolKeepCount = 5,
    excludeTools = [],
    enableToolCompression = false,
    compressionKeepCount = 2,
    compressionMaxLength = 1000,
    logPrefix = "",
    originalRequest,
    preserveKeywords = [],
  } = options;

  let processed = messages;

  // Step 1: Filter orphaned tool results (pre-trim)
  // Removes ToolMessages that have no matching AIMessage with tool_use
  processed = filterOrphanedToolResults(processed, logPrefix);

  // Step 2: Repair dangling tool calls (pre-trim)
  // Creates synthetic ToolMessages for AIMessages with unresolved tool_calls
  // This follows LangGraph Deep Agents "Dangling Tool Call Repair" pattern
  processed = repairDanglingToolCalls(processed, logPrefix);

  // Step 3: Compress verbose tool results (if enabled)
  // This dramatically reduces token usage for tools like getAvailableTemplates,
  // generateAIImage, etc. that return large JSON payloads
  if (enableToolCompression) {
    processed = compressToolResults(processed, {
      keepFullResultsCount: compressionKeepCount,
      maxResultLength: compressionMaxLength,
      logPrefix,
    });
  }

  // Step 4: Clear old tool results (if enabled)
  if (enableToolClearing) {
    processed = clearOldToolResults(processed, {
      keepCount: toolKeepCount,
      excludeTools,
      logPrefix,
    });
  }

  // Step 5: Summarize (if enabled)
  if (enableSummarization) {
    processed = await summarizeIfNeeded(processed, {
      triggerTokens: summarizeTriggerTokens,
      keepMessages: summarizeKeepMessages,
      logPrefix,
    });
  }

  // Step 6: Trim to limit with task-critical message preservation
  processed = await trimMessages(processed, {
    maxTokens,
    model,
    fallbackMessageCount,
    logPrefix,
    originalRequest,
    preserveKeywords,
  });

  // Step 7: Filter orphaned tool results AGAIN (post-trim)
  // This catches orphans created by trimMessages() fallback logic
  // when it keeps last N messages without respecting tool_use/tool_result pairs
  processed = filterOrphanedToolResults(processed, logPrefix);

  // Step 8: Repair dangling tool calls AGAIN (post-trim)
  // This catches danglers created when trimming removes ToolMessages
  // but keeps their corresponding AIMessage with tool_calls
  processed = repairDanglingToolCalls(processed, logPrefix);

  return processed;
}

