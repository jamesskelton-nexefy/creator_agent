/**
 * Context Management Utilities
 *
 * Centralized exports for context management functions.
 */

export {
  // Constants
  TOKEN_LIMITS,
  MESSAGE_LIMITS,
  // Helper functions
  getMessageType,
  hasNonEmptyTextContent,
  hasUsableResponse,
  stripThinkingBlocks,
  // Core functions
  filterOrphanedToolResults,
  trimMessages,
  summarizeIfNeeded,
  clearOldToolResults,
  processContext,
  // Types
  type TrimMessagesOptions,
  type SummarizeOptions,
  type ClearToolResultsOptions,
  type ProcessContextOptions,
} from "./context-management";

