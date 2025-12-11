/**
 * Documents Module
 * 
 * Exports document processing and RAG functionality.
 */

export {
  processDocument,
  getFileTypeFromMime,
  isValidDocumentType,
  type ProcessedChunk,
  type ProcessedDocument,
  type DocumentProcessorOptions,
} from "./documentProcessor";

export {
  VectorStore,
  createVectorStore,
  type DocumentMetadata,
  type ChunkWithEmbedding,
  type SearchResult,
  type TextSearchResult,
  type VectorStoreConfig,
} from "./vectorStore";

export {
  DocumentService,
  createDocumentService,
  type UploadDocumentParams,
  type UploadDocumentResult,
} from "./documentService";









