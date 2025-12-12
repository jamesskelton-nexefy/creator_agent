/**
 * Context Management Utilities
 *
 * Centralized exports for context management functions.
 */

export {
  // Constants
  TOKEN_LIMITS,
  MESSAGE_LIMITS,
  TOOLS_WITH_LARGE_ARGS,
  // Helper functions
  getMessageType,
  hasNonEmptyTextContent,
  hasUsableResponse,
  stripThinkingBlocks,
  deduplicateToolUseIds,
  enforceToolResultOrdering,
  // Core functions
  filterOrphanedToolResults,
  repairDanglingToolCalls,
  trimMessages,
  summarizeIfNeeded,
  clearOldToolResults,
  compressToolResults,
  stripLargeToolCallArgs,
  processContext,
  // Types
  type TrimMessagesOptions,
  type SummarizeOptions,
  type ClearToolResultsOptions,
  type CompressToolResultsOptions,
  type ToolCompressionRule,
  type StripLargeToolCallArgsOptions,
  type ProcessContextOptions,
} from "./context-management";


