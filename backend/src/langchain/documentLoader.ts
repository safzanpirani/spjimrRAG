import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { BaseDocumentLoader } from "@langchain/core/document_loaders/base";
import * as fs from "fs/promises";
import * as path from "path";
import { DocumentMetadata } from "../types/rag";

/**
 * SPJIMR Document Loader
 *
 * Custom document loader for processing SPJIMR PGPM documents from markdown files.
 * Handles the processed markdown data and creates properly formatted documents
 * with appropriate metadata for the RAG system.
 */
export class SPJIMRDocumentLoader extends BaseDocumentLoader {
  private documentsPath: string;
  private chunkSize: number;
  private chunkOverlap: number;

  constructor(
    documentsPath: string = path.join(__dirname, "../../../processed"),
    chunkSize: number = 1200,
    chunkOverlap: number = 250
  ) {
    super();
    this.documentsPath = path.resolve(documentsPath);
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
  }

  /**
   * Load all SPJIMR PGPM documents
   */
  async load(): Promise<Document[]> {
    console.log(`[Document Loader] Loading documents from: ${this.documentsPath}`);

    try {
      const documents: Document[] = [];

      // Get all markdown files in the processed data directory
      const files = await this.getAllMarkdownFiles();

      console.log(`[Document Loader] Found ${files.length} markdown files to process`);

      // Process each markdown file
      for (const filePath of files) {
        try {
          const docs = await this.loadSingleMarkdownDocument(filePath);
          documents.push(...docs);
          console.log(`[Document Loader] Processed ${path.basename(filePath)}: ${docs.length} chunks`);
        } catch (error) {
          console.error(`[Document Loader] Error processing ${filePath}:`, error);
          // Continue processing other files
        }
      }

      console.log(`[Document Loader] Total documents loaded: ${documents.length}`);
      return documents;

    } catch (error) {
      console.error("[Document Loader] Error loading documents:", error);
      throw new Error(`Failed to load documents: ${error}`);
    }
  }

  /**
   * Load and process a single markdown document
   */
  async loadSingleMarkdownDocument(filePath: string): Promise<Document[]> {
    const fileName = path.basename(filePath, '.md');
    const documentType = this.getDocumentType(fileName);

    console.log(`[Document Loader] Processing ${fileName} as ${documentType}`);

    // Read markdown file
    const markdownContent = await fs.readFile(filePath, 'utf-8');

    // Clean and preprocess text (markdown is already clean)
    const cleanedText = this.preprocessMarkdown(markdownContent);

    if (cleanedText.length < 100) {
      console.warn(`[Document Loader] Document ${fileName} has very little content, skipping`);
      return [];
    }

    console.log(`[Document Loader] Loaded ${cleanedText.length} characters from ${fileName}`);

    // Create base metadata
    const baseMetadata: DocumentMetadata = {
      source: fileName,
      program: "PGPM",
      type: documentType,
      lastUpdated: new Date().toISOString()
    };

    // Split document into chunks with markdown-aware separators
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
      separators: ["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    });

    const chunks = await textSplitter.splitText(cleanedText);

    // Create Document objects with metadata
    const documents = chunks.map((chunk, index) => {
      const metadata = {
        ...baseMetadata,
        chunkIndex: index,
        chunkId: `${fileName}_chunk_${index}`
      };

      return new Document({
        pageContent: chunk.trim(),
        metadata: metadata
      });
    });

    return documents.filter(doc => doc.pageContent.length > 50); // Filter out very small chunks
  }

  /**
   * Get all markdown files from the documents directory
   */
  private async getAllMarkdownFiles(): Promise<string[]> {
    const files: string[] = [];

    try {
      // Check main directory for markdown files
      const mainFiles = await fs.readdir(this.documentsPath);
      for (const file of mainFiles) {
        if (file.toLowerCase().endsWith('.md')) {
          files.push(path.join(this.documentsPath, file));
        }
      }

    } catch (error) {
      console.error("[Document Loader] Error reading directory:", error);
      throw new Error(`Cannot access documents directory: ${this.documentsPath}`);
    }

    return files;
  }

  /**
   * Determine document type based on filename
   */
  private getDocumentType(fileName: string): DocumentMetadata['type'] {
    const name = fileName.toLowerCase();

    if (name.includes('admission')) return 'admissions';
    if (name.includes('fee')) return 'fees';
    if (name.includes('placement')) return 'placements';
    if (name.includes('curriculum')) return 'curriculum';
    if (name.includes('eligib')) return 'eligibility';

    return 'general';
  }

  /**
   * Preprocess markdown content (minimal processing since markdown is already clean)
   */
  private preprocessMarkdown(text: string): string {
    return text
      // Normalize line endings and trim trailing spaces per line
      .replace(/\r\n?/g, '\n')
      .replace(/[ \t]+\n/g, '\n')
      // Collapse excessive blank lines but preserve paragraph and heading structure
      .replace(/\n{3,}/g, '\n\n')
      // Keep markdown syntax intact (no aggressive space collapsing)
      .trim();
  }

  /**
   * Preprocess and clean text content
   */
  private preprocessText(text: string): string {
    return text
      // Remove excessive whitespace
      .replace(/\s+/g, ' ')
      // Remove page numbers and headers/footers patterns
      .replace(/Page \d+ of \d+/gi, '')
      .replace(/^\d+\s*$/gm, '')
      // Remove email addresses and URLs (unless important)
      .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '')
      // Clean up special characters but preserve meaning
      .replace(/[^\w\s\-.,!?:;()\[\]]/g, ' ')
      // Remove excessive periods
      .replace(/\.{3,}/g, '...')
      // Normalize spaces
      .replace(/\s+/g, ' ')
      // Trim
      .trim();
  }
}

/**
 * Enhanced text splitter specifically for SPJIMR documents
 */
// Removed unused SPJIMRTextSplitter class

/**
 * Document processing utilities
 */
export class DocumentProcessor {
  /**
   * Validate document content quality
   */
  static validateDocument(doc: Document): boolean {
    const content = doc.pageContent.trim();

    // Basic quality checks
    if (content.length < 50) return false;
    if (content.split(' ').length < 10) return false;

    // Check for meaningful content (not just numbers/symbols)
    const meaningfulWords = content.match(/[a-zA-Z]{3,}/g);
    if (!meaningfulWords || meaningfulWords.length < 5) return false;

    return true;
  }

  /**
   * Deduplicate similar documents
   */
  static deduplicateDocuments(docs: Document[]): Document[] {
    const uniqueDocs: Document[] = [];
    const seen = new Set<string>();

    for (const doc of docs) {
      // Create a simple hash based on first 200 characters
      const hash = doc.pageContent.substring(0, 200).replace(/\s+/g, ' ').trim();

      if (!seen.has(hash)) {
        seen.add(hash);
        uniqueDocs.push(doc);
      }
    }

    console.log(`[Document Processor] Deduplicated ${docs.length} -> ${uniqueDocs.length} documents`);
    return uniqueDocs;
  }

  /**
   * Enhance documents with additional metadata
   */
  static enhanceDocuments(docs: Document[]): Document[] {
    return docs.map(doc => {
      const enhanced = { ...doc };
      const content = doc.pageContent;

      // Add content statistics
      enhanced.metadata = {
        ...doc.metadata,
        wordCount: content.split(/\s+/).length,
        charCount: content.length,
        hasNumbers: /\d/.test(content),
        hasDates: /\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b/.test(content),
        hasPercentages: /\d+%/.test(content),
        hasCurrency: /Rs\.?\s?\d+|₹\s?\d+/.test(content),
        processedAt: new Date().toISOString()
      };

      return enhanced;
    });
  }

  /**
   * Filter documents by content type
   */
  static filterDocumentsByType(
    docs: Document[],
    type: DocumentMetadata['type']
  ): Document[] {
    return docs.filter(doc => doc.metadata['type'] === type);
  }

  /**
   * Get document statistics
   */
  static getDocumentStats(docs: Document[]) {
    const stats = {
      total: docs.length,
      types: {} as Record<string, number>,
      totalWords: 0,
      totalChars: 0,
      avgWordsPerDoc: 0,
      avgCharsPerDoc: 0
    };

    docs.forEach(doc => {
      const type = doc.metadata['type'] as string || 'unknown';
      stats.types[type] = (stats.types[type] || 0) + 1;

      const words = doc.pageContent.split(/\s+/).length;
      const chars = doc.pageContent.length;

      stats.totalWords += words;
      stats.totalChars += chars;
    });

    stats.avgWordsPerDoc = Math.round(stats.totalWords / docs.length);
    stats.avgCharsPerDoc = Math.round(stats.totalChars / docs.length);

    return stats;
  }
}

// Export convenience function for loading all SPJIMR documents
export const loadSPJIMRDocuments = async (documentsPath?: string): Promise<Document[]> => {
  const loader = new SPJIMRDocumentLoader(documentsPath);
  const rawDocs = await loader.load();

  // Process and enhance documents
  const validDocs = rawDocs.filter(DocumentProcessor.validateDocument);
  const uniqueDocs = DocumentProcessor.deduplicateDocuments(validDocs);
  const enhancedDocs = DocumentProcessor.enhanceDocuments(uniqueDocs);

  console.log("[Document Loader] Document processing complete:");
  console.log(DocumentProcessor.getDocumentStats(enhancedDocs));

  return enhancedDocs;
};
