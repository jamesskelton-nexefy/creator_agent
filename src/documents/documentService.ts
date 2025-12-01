/**
 * Document Service
 * 
 * Orchestrates document upload, processing, and storage workflow.
 */

import { processDocument, getFileTypeFromMime, isValidDocumentType } from "./documentProcessor";
import { VectorStore, createVectorStore, DocumentMetadata } from "./vectorStore";

export interface UploadDocumentParams {
  buffer: Buffer;
  filename: string;
  mimeType: string;
  orgId: string;
  projectId?: string;
  category: "course_content" | "framework_content";
  title?: string;
  uploadedBy?: string;
}

export interface UploadDocumentResult {
  documentId: string;
  title: string;
  totalLines: number;
  totalChunks: number;
  status: "ready" | "error";
  errorMessage?: string;
}

export class DocumentService {
  private vectorStore: VectorStore;

  constructor(vectorStore?: VectorStore) {
    this.vectorStore = vectorStore || createVectorStore();
  }

  /**
   * Upload and process a document
   */
  async uploadDocument(params: UploadDocumentParams): Promise<UploadDocumentResult> {
    console.log(`[DocumentService] Starting upload for: ${params.filename}`);

    // Validate file type
    if (!isValidDocumentType(params.mimeType)) {
      throw new Error(`Invalid file type: ${params.mimeType}. Only PDF and DOCX files are supported.`);
    }

    const fileType = getFileTypeFromMime(params.mimeType)!;

    // Generate storage path
    const timestamp = Date.now();
    const sanitizedFilename = params.filename.replace(/[^a-zA-Z0-9.-]/g, "_");
    const storagePath = `${params.orgId}/${timestamp}-${sanitizedFilename}`;

    let documentId: string | undefined;

    try {
      // 1. Upload file to storage
      console.log(`[DocumentService] Uploading file to storage: ${storagePath}`);
      await this.vectorStore.uploadFile(params.buffer, storagePath, params.mimeType);

      // 2. Process document (extract text, chunk)
      console.log(`[DocumentService] Processing document...`);
      const processed = await processDocument(params.buffer, params.filename, fileType);

      // 3. Create document record
      console.log(`[DocumentService] Creating document record...`);
      const document = await this.vectorStore.createDocument({
        orgId: params.orgId,
        projectId: params.projectId,
        title: params.title || processed.title,
        originalFilename: params.filename,
        fileType,
        category: params.category,
        storagePath,
        totalLines: processed.totalLines,
        totalChunks: processed.totalChunks,
        fileSize: params.buffer.length,
        uploadedBy: params.uploadedBy,
      });

      documentId = document.id;

      // 4. Generate embeddings and store chunks
      console.log(`[DocumentService] Storing ${processed.chunks.length} chunks with embeddings...`);
      await this.vectorStore.storeChunks(document.id, processed.chunks);

      // 5. Update status to ready
      await this.vectorStore.updateDocumentStatus(document.id, "ready");

      console.log(`[DocumentService] Document processed successfully: ${document.id}`);

      return {
        documentId: document.id,
        title: document.title,
        totalLines: processed.totalLines,
        totalChunks: processed.totalChunks,
        status: "ready",
      };
    } catch (error) {
      console.error(`[DocumentService] Error processing document:`, error);

      // Update status to error if document was created
      if (documentId) {
        try {
          await this.vectorStore.updateDocumentStatus(
            documentId,
            "error",
            error instanceof Error ? error.message : "Unknown error"
          );
        } catch (updateError) {
          console.error(`[DocumentService] Failed to update error status:`, updateError);
        }
      }

      // Clean up storage if upload succeeded but processing failed
      try {
        await this.vectorStore.deleteFile(storagePath);
      } catch {
        // Ignore cleanup errors
      }

      throw error;
    }
  }

  /**
   * Semantic search across documents
   */
  async searchDocuments(params: {
    query: string;
    limit?: number;
    threshold?: number;
    category?: "course_content" | "framework_content";
    orgId?: string;
    projectId?: string;
  }) {
    return this.vectorStore.searchByEmbedding(params);
  }

  /**
   * Full-text search across documents
   */
  async searchDocumentsByText(params: {
    searchText: string;
    limit?: number;
    category?: "course_content" | "framework_content";
    orgId?: string;
    projectId?: string;
  }) {
    return this.vectorStore.searchByText(params);
  }

  /**
   * Get document lines by range
   */
  async getDocumentLines(params: {
    documentId: string;
    startLine: number;
    numLines: number;
  }) {
    return this.vectorStore.getDocumentLines(params);
  }

  /**
   * Get document by ID
   */
  async getDocument(documentId: string): Promise<DocumentMetadata | null> {
    return this.vectorStore.getDocument(documentId);
  }

  /**
   * Get document by name
   */
  async getDocumentByName(name: string, orgId?: string): Promise<DocumentMetadata | null> {
    return this.vectorStore.getDocumentByName(name, orgId);
  }

  /**
   * Get full document content (all chunks concatenated)
   */
  async getDocumentContent(documentId: string): Promise<string> {
    const chunks = await this.vectorStore.getDocumentChunks(documentId);
    // Sort by chunk index and join content
    const sortedChunks = chunks.sort((a, b) => a.chunkIndex - b.chunkIndex);
    return sortedChunks.map((c) => c.content).join("\n");
  }

  /**
   * List documents
   */
  async listDocuments(params?: {
    category?: "course_content" | "framework_content";
    orgId?: string;
    projectId?: string;
    limit?: number;
    offset?: number;
  }): Promise<DocumentMetadata[]> {
    return this.vectorStore.listDocuments(params);
  }

  /**
   * Delete a document
   */
  async deleteDocument(documentId: string): Promise<void> {
    const document = await this.vectorStore.getDocument(documentId);
    if (document) {
      // Delete from storage first
      try {
        await this.vectorStore.deleteFile(document.storagePath);
      } catch {
        // Ignore storage errors
      }
      // Then delete from database (cascades to chunks)
      await this.vectorStore.deleteDocument(documentId);
    }
  }
}

/**
 * Create a DocumentService instance
 */
export function createDocumentService(): DocumentService {
  return new DocumentService();
}

