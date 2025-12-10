/**
 * Vector Store Service
 * 
 * Handles embedding generation with OpenAI and storage in Supabase pgvector.
 */

import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import type { ProcessedChunk } from "./documentProcessor";

export interface DocumentMetadata {
  id: string;
  orgId: string;
  projectId: string | null;
  title: string;
  originalFilename: string;
  fileType: "pdf" | "docx";
  category: "course_content" | "framework_content";
  storagePath: string;
  status: "processing" | "ready" | "error";
  totalLines: number;
  totalChunks: number;
  fileSize: number;
  uploadedBy?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ChunkWithEmbedding {
  documentId: string;
  content: string;
  embedding: number[];
  chunkIndex: number;
  startLine: number;
  endLine: number;
  metadata: Record<string, unknown>;
}

export interface SearchResult {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  category: string;
  content: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  similarity: number;
}

export interface TextSearchResult {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  category: string;
  content: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  rank: number;
}

export interface VectorStoreConfig {
  supabaseUrl: string;
  supabaseServiceKey: string;
  openaiApiKey: string;
}

export class VectorStore {
  private supabase: SupabaseClient;
  private embeddings: OpenAIEmbeddings;

  constructor(config: VectorStoreConfig) {
    this.supabase = createClient(config.supabaseUrl, config.supabaseServiceKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false,
      },
    });

    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: config.openaiApiKey,
      modelName: "text-embedding-3-small",
    });
  }

  /**
   * Generate embeddings for text
   */
  async generateEmbedding(text: string): Promise<number[]> {
    const embeddings = await this.embeddings.embedQuery(text);
    return embeddings;
  }

  /**
   * Generate embeddings for multiple texts (batch)
   */
  async generateEmbeddings(texts: string[]): Promise<number[][]> {
    const embeddings = await this.embeddings.embedDocuments(texts);
    return embeddings;
  }

  /**
   * Create a new document record
   */
  async createDocument(params: {
    orgId: string;
    projectId?: string;
    title: string;
    originalFilename: string;
    fileType: "pdf" | "docx";
    category: "course_content" | "framework_content";
    storagePath: string;
    totalLines: number;
    totalChunks: number;
    fileSize: number;
    uploadedBy?: string;
  }): Promise<DocumentMetadata> {
    const { data, error } = await this.supabase
      .schema("documents")
      .from("documents")
      .insert({
        org_id: params.orgId,
        project_id: params.projectId || null,
        title: params.title,
        original_filename: params.originalFilename,
        file_type: params.fileType,
        category: params.category,
        storage_path: params.storagePath,
        status: "processing",
        total_lines: params.totalLines,
        total_chunks: params.totalChunks,
        file_size: params.fileSize,
        uploaded_by: params.uploadedBy || null,
      })
      .select()
      .single();

    if (error) {
      throw new Error(`Failed to create document: ${error.message}`);
    }

    return this.mapDocumentRow(data);
  }

  /**
   * Store chunks with embeddings
   */
  async storeChunks(documentId: string, chunks: ProcessedChunk[]): Promise<void> {
    console.log(`[VectorStore] Generating embeddings for ${chunks.length} chunks...`);

    // Generate embeddings in batches to avoid rate limits
    const batchSize = 10;
    const allChunksWithEmbeddings: ChunkWithEmbedding[] = [];

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      const texts = batch.map((c) => c.content);
      const embeddings = await this.generateEmbeddings(texts);

      for (let j = 0; j < batch.length; j++) {
        allChunksWithEmbeddings.push({
          documentId,
          content: batch[j].content,
          embedding: embeddings[j],
          chunkIndex: batch[j].chunkIndex,
          startLine: batch[j].startLine,
          endLine: batch[j].endLine,
          metadata: batch[j].metadata,
        });
      }

      console.log(`[VectorStore] Processed batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(chunks.length / batchSize)}`);
    }

    console.log(`[VectorStore] Storing ${allChunksWithEmbeddings.length} chunks in database...`);

    // Insert chunks in batches
    for (let i = 0; i < allChunksWithEmbeddings.length; i += batchSize) {
      const batch = allChunksWithEmbeddings.slice(i, i + batchSize);

      const { error } = await this.supabase
        .schema("documents")
        .from("document_chunks")
        .insert(
          batch.map((chunk) => ({
            document_id: chunk.documentId,
            content: chunk.content,
            embedding: JSON.stringify(chunk.embedding),
            chunk_index: chunk.chunkIndex,
            start_line: chunk.startLine,
            end_line: chunk.endLine,
            metadata: chunk.metadata,
          }))
        );

      if (error) {
        throw new Error(`Failed to store chunks: ${error.message}`);
      }
    }

    console.log(`[VectorStore] Successfully stored all chunks`);
  }

  /**
   * Update document status
   */
  async updateDocumentStatus(
    documentId: string,
    status: "processing" | "ready" | "error",
    errorMessage?: string
  ): Promise<void> {
    const updateData: Record<string, unknown> = { status };
    if (errorMessage) {
      updateData.error_message = errorMessage;
    }

    const { error } = await this.supabase
      .schema("documents")
      .from("documents")
      .update(updateData)
      .eq("id", documentId);

    if (error) {
      throw new Error(`Failed to update document status: ${error.message}`);
    }
  }

  /**
   * Semantic search using vector similarity
   */
  async searchByEmbedding(params: {
    query: string;
    limit?: number;
    threshold?: number;
    category?: "course_content" | "framework_content";
    orgId?: string;
    projectId?: string;
  }): Promise<SearchResult[]> {
    const queryEmbedding = await this.generateEmbedding(params.query);

    const { data, error } = await this.supabase.rpc("search_documents_by_embedding", {
      query_embedding: JSON.stringify(queryEmbedding),
      match_threshold: params.threshold ?? 0.5,
      match_count: params.limit ?? 10,
      filter_category: params.category ?? null,
      filter_org_id: params.orgId ?? null,
      filter_project_id: params.projectId ?? null,
    });

    if (error) {
      throw new Error(`Search failed: ${error.message}`);
    }

    return (data || []).map((row: Record<string, unknown>) => ({
      chunkId: row.chunk_id as string,
      documentId: row.document_id as string,
      documentTitle: row.document_title as string,
      category: row.category as string,
      content: row.content as string,
      chunkIndex: row.chunk_index as number,
      startLine: row.start_line as number,
      endLine: row.end_line as number,
      similarity: row.similarity as number,
    }));
  }

  /**
   * Full-text search
   */
  async searchByText(params: {
    searchText: string;
    limit?: number;
    category?: "course_content" | "framework_content";
    orgId?: string;
    projectId?: string;
  }): Promise<TextSearchResult[]> {
    const { data, error } = await this.supabase.rpc("search_documents_by_text", {
      search_text: params.searchText,
      match_count: params.limit ?? 10,
      filter_category: params.category ?? null,
      filter_org_id: params.orgId ?? null,
      filter_project_id: params.projectId ?? null,
    });

    if (error) {
      throw new Error(`Text search failed: ${error.message}`);
    }

    return (data || []).map((row: Record<string, unknown>) => ({
      chunkId: row.chunk_id as string,
      documentId: row.document_id as string,
      documentTitle: row.document_title as string,
      category: row.category as string,
      content: row.content as string,
      chunkIndex: row.chunk_index as number,
      startLine: row.start_line as number,
      endLine: row.end_line as number,
      rank: row.rank as number,
    }));
  }

  /**
   * Get document lines by range
   */
  async getDocumentLines(params: {
    documentId: string;
    startLine: number;
    numLines: number;
  }): Promise<{ content: string; chunkIndex: number; startLine: number; endLine: number }[]> {
    const { data, error } = await this.supabase.rpc("get_document_lines", {
      p_document_id: params.documentId,
      p_start_line: params.startLine,
      p_num_lines: params.numLines,
    });

    if (error) {
      throw new Error(`Failed to get document lines: ${error.message}`);
    }

    return (data || []).map((row: Record<string, unknown>) => ({
      content: row.content as string,
      chunkIndex: row.chunk_index as number,
      startLine: row.start_line as number,
      endLine: row.end_line as number,
    }));
  }

  /**
   * Get document by ID
   */
  async getDocument(documentId: string): Promise<DocumentMetadata | null> {
    const { data, error } = await this.supabase
      .schema("documents")
      .from("documents")
      .select("*")
      .eq("id", documentId)
      .single();

    if (error) {
      if (error.code === "PGRST116") {
        return null;
      }
      throw new Error(`Failed to get document: ${error.message}`);
    }

    return this.mapDocumentRow(data);
  }

  /**
   * Get document by name (title or original filename)
   * Uses case-insensitive partial matching
   */
  async getDocumentByName(
    name: string,
    orgId?: string
  ): Promise<DocumentMetadata | null> {
    // First try exact title match (case-insensitive)
    let query = this.supabase
      .schema("documents")
      .from("documents")
      .select("*")
      .ilike("title", name)
      .eq("status", "ready");

    if (orgId) {
      query = query.eq("org_id", orgId);
    }

    let { data, error } = await query.limit(1).maybeSingle();

    if (error && error.code !== "PGRST116") {
      throw new Error(`Failed to get document by name: ${error.message}`);
    }

    // If found with exact match, return it
    if (data) {
      return this.mapDocumentRow(data);
    }

    // Try partial title match (contains)
    query = this.supabase
      .schema("documents")
      .from("documents")
      .select("*")
      .ilike("title", `%${name}%`)
      .eq("status", "ready");

    if (orgId) {
      query = query.eq("org_id", orgId);
    }

    ({ data, error } = await query.limit(1).maybeSingle());

    if (error && error.code !== "PGRST116") {
      throw new Error(`Failed to get document by name: ${error.message}`);
    }

    if (data) {
      return this.mapDocumentRow(data);
    }

    // Try original filename match
    query = this.supabase
      .schema("documents")
      .from("documents")
      .select("*")
      .ilike("original_filename", `%${name}%`)
      .eq("status", "ready");

    if (orgId) {
      query = query.eq("org_id", orgId);
    }

    ({ data, error } = await query.limit(1).maybeSingle());

    if (error && error.code !== "PGRST116") {
      throw new Error(`Failed to get document by name: ${error.message}`);
    }

    return data ? this.mapDocumentRow(data) : null;
  }

  /**
   * Get all chunks for a document
   */
  async getDocumentChunks(documentId: string): Promise<ProcessedChunk[]> {
    const { data, error } = await this.supabase
      .schema("documents")
      .from("document_chunks")
      .select("*")
      .eq("document_id", documentId)
      .order("chunk_index");

    if (error) {
      throw new Error(`Failed to get document chunks: ${error.message}`);
    }

    return (data || []).map((row: Record<string, unknown>) => ({
      content: row.content as string,
      chunkIndex: row.chunk_index as number,
      startLine: row.start_line as number,
      endLine: row.end_line as number,
      metadata: (row.metadata as Record<string, unknown>) || {},
    }));
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
    let query = this.supabase
      .schema("documents")
      .from("documents")
      .select("*")
      .eq("status", "ready")
      .order("created_at", { ascending: false });

    if (params?.category) {
      query = query.eq("category", params.category);
    }
    if (params?.orgId) {
      query = query.eq("org_id", params.orgId);
    }
    if (params?.projectId) {
      query = query.eq("project_id", params.projectId);
    }
    if (params?.limit) {
      query = query.limit(params.limit);
    }
    if (params?.offset) {
      query = query.range(params.offset, params.offset + (params.limit || 50) - 1);
    }

    const { data, error } = await query;

    if (error) {
      throw new Error(`Failed to list documents: ${error.message}`);
    }

    return (data || []).map(this.mapDocumentRow);
  }

  /**
   * Delete document and all its chunks
   */
  async deleteDocument(documentId: string): Promise<void> {
    const { error } = await this.supabase
      .schema("documents")
      .from("documents")
      .delete()
      .eq("id", documentId);

    if (error) {
      throw new Error(`Failed to delete document: ${error.message}`);
    }
  }

  /**
   * Upload file to storage
   */
  async uploadFile(
    buffer: Buffer,
    storagePath: string,
    mimeType: string
  ): Promise<string> {
    const { error } = await this.supabase.storage
      .from("documents")
      .upload(storagePath, buffer, {
        contentType: mimeType,
        upsert: false,
      });

    if (error) {
      throw new Error(`Failed to upload file: ${error.message}`);
    }

    return storagePath;
  }

  /**
   * Delete file from storage
   */
  async deleteFile(storagePath: string): Promise<void> {
    const { error } = await this.supabase.storage
      .from("documents")
      .remove([storagePath]);

    if (error) {
      throw new Error(`Failed to delete file: ${error.message}`);
    }
  }

  /**
   * Map database row to DocumentMetadata
   */
  private mapDocumentRow(row: Record<string, unknown>): DocumentMetadata {
    return {
      id: row.id as string,
      orgId: row.org_id as string,
      projectId: row.project_id as string | null,
      title: row.title as string,
      originalFilename: row.original_filename as string,
      fileType: row.file_type as "pdf" | "docx",
      category: row.category as "course_content" | "framework_content",
      storagePath: row.storage_path as string,
      status: row.status as "processing" | "ready" | "error",
      totalLines: row.total_lines as number,
      totalChunks: row.total_chunks as number,
      fileSize: row.file_size as number,
      uploadedBy: row.uploaded_by as string | undefined,
      createdAt: row.created_at as string,
      updatedAt: row.updated_at as string,
    };
  }
}

/**
 * Create a VectorStore instance from environment variables
 */
export function createVectorStore(): VectorStore {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const openaiApiKey = process.env.OPENAI_API_KEY;

  if (!supabaseUrl) {
    throw new Error("SUPABASE_URL environment variable is required");
  }
  if (!supabaseServiceKey) {
    throw new Error("SUPABASE_SERVICE_ROLE_KEY environment variable is required");
  }
  if (!openaiApiKey) {
    throw new Error("OPENAI_API_KEY environment variable is required");
  }

  return new VectorStore({
    supabaseUrl,
    supabaseServiceKey,
    openaiApiKey,
  });
}








