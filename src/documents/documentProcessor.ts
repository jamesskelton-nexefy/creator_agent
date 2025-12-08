/**
 * Document Processor
 * 
 * Handles loading and chunking PDF/DOCX documents using LangChain loaders.
 * Tracks line numbers for each chunk to enable line-based retrieval.
 */

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

export interface ProcessedChunk {
  content: string;
  chunkIndex: number;
  startLine: number;
  endLine: number;
  metadata: Record<string, unknown>;
}

export interface ProcessedDocument {
  title: string;
  originalFilename: string;
  fileType: "pdf" | "docx";
  totalLines: number;
  totalChunks: number;
  chunks: ProcessedChunk[];
  rawContent: string;
}

export interface DocumentProcessorOptions {
  chunkSize?: number;
  chunkOverlap?: number;
}

const DEFAULT_OPTIONS: Required<DocumentProcessorOptions> = {
  chunkSize: 1000,
  chunkOverlap: 200,
};

/**
 * Load PDF document from buffer
 */
async function loadPdfFromBuffer(buffer: Buffer, filename: string): Promise<Document[]> {
  // Dynamic import to handle ESM/CJS compatibility
  const { PDFLoader } = await import("@langchain/community/document_loaders/fs/pdf");
  const { Blob } = await import("buffer");
  
  // Create a Blob from the buffer for PDFLoader
  const blob = new Blob([buffer], { type: "application/pdf" });
  
  const loader = new PDFLoader(blob as unknown as Blob, {
    splitPages: true,
  });
  
  const docs = await loader.load();
  
  // Add filename to metadata
  return docs.map(doc => ({
    ...doc,
    metadata: {
      ...doc.metadata,
      source: filename,
    },
  }));
}

/**
 * Load DOCX document from buffer
 */
async function loadDocxFromBuffer(buffer: Buffer, filename: string): Promise<Document[]> {
  // Use mammoth for DOCX parsing
  const mammoth = await import("mammoth");
  
  const result = await mammoth.extractRawText({ buffer });
  const text = result.value;
  
  return [{
    pageContent: text,
    metadata: {
      source: filename,
    },
  }];
}

/**
 * Count lines in text content
 */
function countLines(text: string): number {
  return text.split("\n").length;
}

/**
 * Calculate line numbers for a chunk within the full document
 */
function calculateLineNumbers(
  fullContent: string,
  chunkContent: string,
  searchStartIndex: number
): { startLine: number; endLine: number; foundIndex: number } {
  // Find where this chunk appears in the full content
  const foundIndex = fullContent.indexOf(chunkContent, searchStartIndex);
  
  if (foundIndex === -1) {
    // Chunk not found exactly, try to find approximate location
    // This can happen with overlapping chunks
    const beforeChunk = fullContent.substring(0, searchStartIndex);
    const startLine = countLines(beforeChunk);
    const endLine = startLine + countLines(chunkContent) - 1;
    return { startLine, endLine, foundIndex: searchStartIndex };
  }
  
  // Count lines before this chunk
  const beforeChunk = fullContent.substring(0, foundIndex);
  const startLine = countLines(beforeChunk);
  const endLine = startLine + countLines(chunkContent) - 1;
  
  return { startLine, endLine, foundIndex: foundIndex + chunkContent.length };
}

/**
 * Process a document buffer into chunks with embeddings-ready format
 */
export async function processDocument(
  buffer: Buffer,
  filename: string,
  fileType: "pdf" | "docx",
  options: DocumentProcessorOptions = {}
): Promise<ProcessedDocument> {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  console.log(`[DocumentProcessor] Processing ${fileType.toUpperCase()}: ${filename}`);
  
  // Load document based on type
  let docs: Document[];
  
  if (fileType === "pdf") {
    docs = await loadPdfFromBuffer(buffer, filename);
  } else if (fileType === "docx") {
    docs = await loadDocxFromBuffer(buffer, filename);
  } else {
    throw new Error(`Unsupported file type: ${fileType}`);
  }
  
  // Combine all pages/sections into single content
  const rawContent = docs.map(doc => doc.pageContent).join("\n\n");
  const totalLines = countLines(rawContent);
  
  console.log(`[DocumentProcessor] Loaded document with ${totalLines} lines`);
  
  // Create text splitter
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: opts.chunkSize,
    chunkOverlap: opts.chunkOverlap,
    separators: ["\n\n", "\n", ". ", " ", ""],
  });
  
  // Split into chunks
  const splitDocs = await textSplitter.splitDocuments([
    new Document({
      pageContent: rawContent,
      metadata: { source: filename },
    }),
  ]);
  
  console.log(`[DocumentProcessor] Split into ${splitDocs.length} chunks`);
  
  // Calculate line numbers for each chunk
  let searchStartIndex = 0;
  const chunks: ProcessedChunk[] = splitDocs.map((doc, index) => {
    const lineInfo = calculateLineNumbers(rawContent, doc.pageContent, searchStartIndex);
    searchStartIndex = lineInfo.foundIndex;
    
    return {
      content: doc.pageContent,
      chunkIndex: index,
      startLine: lineInfo.startLine,
      endLine: lineInfo.endLine,
      metadata: {
        ...doc.metadata,
        chunkIndex: index,
      },
    };
  });
  
  // Generate title from filename (remove extension)
  const title = filename.replace(/\.(pdf|docx)$/i, "");
  
  return {
    title,
    originalFilename: filename,
    fileType,
    totalLines,
    totalChunks: chunks.length,
    chunks,
    rawContent,
  };
}

/**
 * Get file type from MIME type
 */
export function getFileTypeFromMime(mimeType: string): "pdf" | "docx" | null {
  if (mimeType === "application/pdf") {
    return "pdf";
  }
  if (mimeType === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
    return "docx";
  }
  return null;
}

/**
 * Validate file type
 */
export function isValidDocumentType(mimeType: string): boolean {
  return getFileTypeFromMime(mimeType) !== null;
}


