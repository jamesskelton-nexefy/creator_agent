/**
 * Deep Agents TypeScript Implementation
 *
 * A TypeScript port of the Python Deep Agents library for building controllable AI agents with LangGraph.
 * This implementation maintains 1:1 compatibility with the Python version.
 */

export { createDeepAgent, type CreateDeepAgentParams } from "./agent.js";

// Export middleware
export {
  createFilesystemMiddleware,
  createSubAgentMiddleware,
  createPatchToolCallsMiddleware,
  createCopilotKitMiddleware,
  type FilesystemMiddlewareOptions,
  type SubAgentMiddlewareOptions,
  type CopilotKitMiddlewareOptions,
  type CopilotKitState,
  type PlannedNode,
  type SubAgent,
  type FileData,
} from "./middleware/index.js";

// Export backends
export {
  StateBackend,
  StoreBackend,
  FilesystemBackend,
  CompositeBackend,
  type BackendProtocol,
  type BackendFactory,
  type FileInfo,
  type GrepMatch,
  type WriteResult,
  type EditResult,
} from "./backends/index.js";

// Export documents module
export {
  DocumentService,
  createDocumentService,
  VectorStore,
  createVectorStore,
  processDocument,
  getFileTypeFromMime,
  isValidDocumentType,
  type DocumentMetadata,
  type ProcessedChunk,
  type ProcessedDocument,
  type SearchResult,
  type TextSearchResult,
  type UploadDocumentParams,
  type UploadDocumentResult,
} from "./documents/index.js";
