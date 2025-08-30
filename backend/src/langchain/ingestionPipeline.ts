import { Document } from "@langchain/core/documents";
import { SPJIMRDocumentLoader } from "./documentLoader";
import { SPJIMRVectorStore, getVectorStore } from "./vectorStore";
import { DocumentMetadata } from "../types/rag";
import * as fs from "fs/promises";
import * as path from "path";

/**
 * Document Ingestion Pipeline for SPJIMR PGPM RAG System
 *
 * Handles the complete pipeline from raw documents to indexed vectors:
 * 1. Discovery of markdown files in /processed/ directory
 * 2. Document loading and processing
 * 3. Text chunking and metadata enhancement
 * 4. Vector embedding and indexing
 * 5. Progress monitoring and error handling
 */
export class DocumentIngestionPipeline {
  private documentLoader: SPJIMRDocumentLoader;
  private vectorStore: SPJIMRVectorStore | null = null;
  private readonly preprocessedDataPath: string;
  private readonly batchSize: number = 50;
  private readonly maxRetries: number = 3;

  // Progress tracking
  private totalFiles: number = 0;
  private processedFiles: number = 0;
  private failedFiles: string[] = [];
  private totalDocuments: number = 0;
  private indexedDocuments: number = 0;

  constructor(preprocessedDataPath?: string) {
    this.preprocessedDataPath = preprocessedDataPath ||
      path.join(__dirname, "../../../processed");

    this.documentLoader = new SPJIMRDocumentLoader(this.preprocessedDataPath);

    console.log(`[Ingestion Pipeline] Initialized with data path: ${this.preprocessedDataPath}`);
  }

  /**
   * Initialize the ingestion pipeline
   */
  async initialize(): Promise<void> {
    try {
      console.log("[Ingestion Pipeline] Initializing pipeline...");

      // Initialize vector store
      this.vectorStore = await getVectorStore();

      // Verify preprocessed data directory exists
      await this.verifyDataDirectory();

      console.log("[Ingestion Pipeline] Pipeline initialized successfully");
    } catch (error) {
      console.error("[Ingestion Pipeline] Initialization failed:", error);
      throw new Error(`Pipeline initialization failed: ${error}`);
    }
  }

  /**
   * Verify that the preprocessed data directory exists and is accessible
   */
  private async verifyDataDirectory(): Promise<void> {
    try {
      const stats = await fs.stat(this.preprocessedDataPath);

      if (!stats.isDirectory()) {
        throw new Error(`Path is not a directory: ${this.preprocessedDataPath}`);
      }

      // Test read access
      await fs.readdir(this.preprocessedDataPath);

      console.log(`[Ingestion Pipeline] Data directory verified: ${this.preprocessedDataPath}`);
    } catch (error) {
      if ((error as any).code === 'ENOENT') {
        throw new Error(`Preprocessed data directory not found: ${this.preprocessedDataPath}`);
      }
      throw new Error(`Cannot access data directory: ${error}`);
    }
  }

  /**
   * Discover all markdown files in the preprocessed data directory
   */
  async discoverMarkdownFiles(): Promise<string[]> {
    console.log("[Ingestion Pipeline] Discovering markdown files...");

    try {
      const files = await this.walkDirectory(this.preprocessedDataPath);
      const markdownFiles = files.filter(file => file.toLowerCase().endsWith('.md'));

      console.log(`[Ingestion Pipeline] Found ${markdownFiles.length} markdown files`);

      if (markdownFiles.length === 0) {
        console.warn("[Ingestion Pipeline] No markdown files found in preprocessed data directory");
      }

      return markdownFiles;
    } catch (error) {
      console.error("[Ingestion Pipeline] Error discovering markdown files:", error);
      throw new Error(`Markdown discovery failed: ${error}`);
    }
  }

  /**
   * Recursively walk directory to find all files
   */
  private async walkDirectory(dir: string): Promise<string[]> {
    const files: string[] = [];

    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          const subFiles = await this.walkDirectory(fullPath);
          files.push(...subFiles);
        } else if (entry.isFile()) {
          files.push(fullPath);
        }
      }
    } catch (error) {
      console.warn(`[Ingestion Pipeline] Could not read directory ${dir}:`, error);
    }

    return files;
  }

  /**
   * Process a single markdown file
   */
  async processMarkdownFile(filePath: string): Promise<Document[]> {
    const fileName = path.basename(filePath);
    console.log(`[Ingestion Pipeline] Processing file: ${fileName}`);

    let retryCount = 0;

    while (retryCount < this.maxRetries) {
      try {
        // Create a file-specific document loader
        const fileSpecificLoader = new SPJIMRDocumentLoader();
        
        // Load just this specific file
        const documents = await fileSpecificLoader.loadSingleMarkdownDocument(filePath);

        if (documents.length === 0) {
          console.warn(`[Ingestion Pipeline] No documents extracted from ${fileName}`);
          return [];
        }

        // Enhance metadata with file information
        const enhancedDocuments = documents.map((doc: any) => {
          const metadata: DocumentMetadata = {
            ...doc.metadata,
            sourceFile: fileName,
            filePath: path.relative(this.preprocessedDataPath, filePath),
            processedAt: new Date().toISOString(),
            version: "1.0"
          };

          return new Document({
            pageContent: doc.pageContent,
            metadata
          });
        });

        console.log(`[Ingestion Pipeline] Processed ${fileName}: ${enhancedDocuments.length} chunks`);
        return enhancedDocuments;

      } catch (error) {
        retryCount++;
        console.warn(`[Ingestion Pipeline] Error processing ${fileName} (attempt ${retryCount}):`, error);

        if (retryCount >= this.maxRetries) {
          console.error(`[Ingestion Pipeline] Failed to process ${fileName} after ${this.maxRetries} attempts`);
          this.failedFiles.push(filePath);
          return [];
        }

        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000 * retryCount));
      }
    }

    return [];
  }

  /**
   * Index documents in the vector store in batches
   */
  async indexDocuments(documents: Document[]): Promise<void> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    if (documents.length === 0) {
      console.warn("[Ingestion Pipeline] No documents to index");
      return;
    }

    console.log(`[Ingestion Pipeline] Indexing ${documents.length} documents...`);

    // Process documents in batches
    for (let i = 0; i < documents.length; i += this.batchSize) {
      const batch = documents.slice(i, i + this.batchSize);
      const batchNumber = Math.floor(i / this.batchSize) + 1;
      const totalBatches = Math.ceil(documents.length / this.batchSize);

      console.log(`[Ingestion Pipeline] Indexing batch ${batchNumber}/${totalBatches} (${batch.length} documents)`);

      try {
        await this.vectorStore.addDocuments(batch);
        this.indexedDocuments += batch.length;

        // Brief pause between batches to avoid overwhelming the system
        if (i + this.batchSize < documents.length) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }

      } catch (error) {
        console.error(`[Ingestion Pipeline] Error indexing batch ${batchNumber}:`, error);
        throw new Error(`Batch indexing failed: ${error}`);
      }
    }

    console.log(`[Ingestion Pipeline] Successfully indexed ${this.indexedDocuments} documents`);
  }

  /**
   * Run the complete ingestion pipeline
   */
  async runPipeline(options: {
    forceReindex?: boolean;
    filePattern?: RegExp;
    maxFiles?: number;
  } = {}): Promise<{
    success: boolean;
    totalFiles: number;
    processedFiles: number;
    failedFiles: string[];
    totalDocuments: number;
    indexedDocuments: number;
    duration: number;
  }> {
    const startTime = Date.now();

    console.log("[Ingestion Pipeline] Starting document ingestion pipeline...");

    try {
      // Initialize if not already done
      if (!this.vectorStore) {
        await this.initialize();
      }

      // Check existing documents unless force reindex
      if (!options.forceReindex) {
        const existingCount = await this.vectorStore!.getDocumentCount();
        if (existingCount > 0) {
          console.log(`[Ingestion Pipeline] Found ${existingCount} existing documents. Use forceReindex=true to reprocess.`);
          return {
            success: true,
            totalFiles: 0,
            processedFiles: 0,
            failedFiles: [],
            totalDocuments: existingCount,
            indexedDocuments: 0,
            duration: Date.now() - startTime
          };
        }
      } else {
        console.log("[Ingestion Pipeline] Force reindex enabled, clearing existing documents...");
        await this.vectorStore!.deleteDocuments({ program: "PGPM" });
      }

      // Discover markdown files
      let markdownFiles = await this.discoverMarkdownFiles();

      // Apply file pattern filter if provided
      if (options.filePattern) {
        markdownFiles = markdownFiles.filter(file => options.filePattern!.test(file));
        console.log(`[Ingestion Pipeline] Filtered to ${markdownFiles.length} files matching pattern`);
      }

      // Limit files if specified
      if (options.maxFiles && options.maxFiles > 0) {
        markdownFiles = markdownFiles.slice(0, options.maxFiles);
        console.log(`[Ingestion Pipeline] Limited to first ${markdownFiles.length} files`);
      }

      this.totalFiles = markdownFiles.length;

      if (this.totalFiles === 0) {
        console.warn("[Ingestion Pipeline] No files to process");
        return {
          success: true,
          totalFiles: 0,
          processedFiles: 0,
          failedFiles: [],
          totalDocuments: 0,
          indexedDocuments: 0,
          duration: Date.now() - startTime
        };
      }

      // Process files and collect documents
      const allDocuments: Document[] = [];

      for (let i = 0; i < markdownFiles.length; i++) {
        const filePath = markdownFiles[i];
        if (!filePath) continue;
        
        const fileName = path.basename(filePath);

        console.log(`[Ingestion Pipeline] Processing file ${i + 1}/${this.totalFiles}: ${fileName}`);

        try {
          const documents = await this.processMarkdownFile(filePath);
          allDocuments.push(...documents);
          this.processedFiles++;
          this.totalDocuments += documents.length;

          // Progress update
          const progress = ((i + 1) / this.totalFiles * 100).toFixed(1);
          console.log(`[Ingestion Pipeline] Progress: ${progress}% (${this.processedFiles}/${this.totalFiles} files)`);

        } catch (error) {
          console.error(`[Ingestion Pipeline] Failed to process ${fileName}:`, error);
          this.failedFiles.push(filePath);
        }
      }

      console.log(`[Ingestion Pipeline] Document processing complete: ${allDocuments.length} total documents`);

      // Index all documents
      if (allDocuments.length > 0) {
        await this.indexDocuments(allDocuments);
      }

      const duration = Date.now() - startTime;
      const durationSeconds = (duration / 1000).toFixed(1);

      console.log("[Ingestion Pipeline] Pipeline completed successfully");
      console.log(`[Ingestion Pipeline] Summary:`);
      console.log(`  - Files processed: ${this.processedFiles}/${this.totalFiles}`);
      console.log(`  - Failed files: ${this.failedFiles.length}`);
      console.log(`  - Total documents: ${this.totalDocuments}`);
      console.log(`  - Indexed documents: ${this.indexedDocuments}`);
      console.log(`  - Duration: ${durationSeconds}s`);

      if (this.failedFiles.length > 0) {
        console.warn("[Ingestion Pipeline] Failed files:");
        this.failedFiles.forEach(file => console.warn(`  - ${path.basename(file)}`));
      }

      return {
        success: true,
        totalFiles: this.totalFiles,
        processedFiles: this.processedFiles,
        failedFiles: this.failedFiles,
        totalDocuments: this.totalDocuments,
        indexedDocuments: this.indexedDocuments,
        duration
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      console.error("[Ingestion Pipeline] Pipeline failed:", error);

      return {
        success: false,
        totalFiles: this.totalFiles,
        processedFiles: this.processedFiles,
        failedFiles: this.failedFiles,
        totalDocuments: this.totalDocuments,
        indexedDocuments: this.indexedDocuments,
        duration
      };
    }
  }

  /**
   * Incremental update: process only new or modified files
   */
  async runIncrementalUpdate(): Promise<{
    success: boolean;
    newFiles: string[];
    processedFiles: number;
    newDocuments: number;
  }> {
    console.log("[Ingestion Pipeline] Running incremental update...");

    try {
      if (!this.vectorStore) {
        await this.initialize();
      }

      // Get existing files from vector store metadata
      const existingStats = await this.vectorStore!.getStatistics();
      console.log("[Ingestion Pipeline] Current statistics:", existingStats);

      // Discover current files
      const currentFiles = await this.discoverMarkdownFiles();

      // For now, we'll process all files since we don't have modification time tracking
      // In a production system, you'd compare modification times or use file hashes
      console.log(`[Ingestion Pipeline] Found ${currentFiles.length} files for incremental processing`);

      // This is a simplified version - in production you'd implement proper change detection
      if (existingStats.totalDocuments === 0) {
        console.log("[Ingestion Pipeline] No existing documents, running full pipeline...");
        const result = await this.runPipeline();
        return {
          success: result.success,
          newFiles: currentFiles,
          processedFiles: result.processedFiles,
          newDocuments: result.indexedDocuments
        };
      } else {
        console.log("[Ingestion Pipeline] Documents already exist, skipping incremental update");
        return {
          success: true,
          newFiles: [],
          processedFiles: 0,
          newDocuments: 0
        };
      }

    } catch (error) {
      console.error("[Ingestion Pipeline] Incremental update failed:", error);
      return {
        success: false,
        newFiles: [],
        processedFiles: 0,
        newDocuments: 0
      };
    }
  }

  /**
   * Validate the indexed data
   */
  async validateIndexedData(): Promise<{
    valid: boolean;
    issues: string[];
    statistics: any;
  }> {
    console.log("[Ingestion Pipeline] Validating indexed data...");

    const issues: string[] = [];

    try {
      if (!this.vectorStore) {
        await this.initialize();
      }

      // Get statistics
      const stats = await this.vectorStore!.getStatistics();

      // Check if we have documents
      if (stats.totalDocuments === 0) {
        issues.push("No documents found in vector store");
      }

      // Check document types
      const expectedTypes = ['syllabus', 'handbook', 'admission', 'curriculum'];
      const foundTypes = Object.keys(stats.documentsByType);

      for (const expectedType of expectedTypes) {
        if (!foundTypes.includes(expectedType)) {
          issues.push(`Missing document type: ${expectedType}`);
        }
      }

      // Test similarity search
      try {
        const testQuery = "What is SPJIMR PGPM?";
        const testResults = await this.vectorStore!.similaritySearch(testQuery, 1);

        if (testResults.length === 0) {
          issues.push("Similarity search returned no results for test query");
        }
      } catch (error) {
        issues.push(`Similarity search failed: ${error}`);
      }

      // Check vector store health
      const health = await this.vectorStore!.healthCheck();

      if (health.status !== 'healthy') {
        issues.push(`Vector store health check failed: ${JSON.stringify(health.details)}`);
      }

      const valid = issues.length === 0;

      console.log(`[Ingestion Pipeline] Validation ${valid ? 'passed' : 'failed'}`);
      if (issues.length > 0) {
        console.log("[Ingestion Pipeline] Issues found:");
        issues.forEach(issue => console.log(`  - ${issue}`));
      }

      return {
        valid,
        issues,
        statistics: stats
      };

    } catch (error) {
      console.error("[Ingestion Pipeline] Validation error:", error);
      return {
        valid: false,
        issues: [`Validation error: ${error}`],
        statistics: {}
      };
    }
  }

  /**
   * Get current progress information
   */
  getProgress(): {
    totalFiles: number;
    processedFiles: number;
    failedFiles: number;
    totalDocuments: number;
    indexedDocuments: number;
    progressPercentage: number;
  } {
    const progressPercentage = this.totalFiles > 0
      ? (this.processedFiles / this.totalFiles) * 100
      : 0;

    return {
      totalFiles: this.totalFiles,
      processedFiles: this.processedFiles,
      failedFiles: this.failedFiles.length,
      totalDocuments: this.totalDocuments,
      indexedDocuments: this.indexedDocuments,
      progressPercentage: Math.round(progressPercentage * 100) / 100
    };
  }

  /**
   * Reset progress counters
   */
  resetProgress(): void {
    this.totalFiles = 0;
    this.processedFiles = 0;
    this.failedFiles = [];
    this.totalDocuments = 0;
    this.indexedDocuments = 0;
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    console.log("[Ingestion Pipeline] Cleaning up...");

    if (this.vectorStore) {
      await this.vectorStore.cleanup();
    }

    this.resetProgress();
    console.log("[Ingestion Pipeline] Cleanup complete");
  }
}

/**
 * Convenience function to run the ingestion pipeline
 */
export async function runIngestionPipeline(options?: {
  forceReindex?: boolean;
  filePattern?: RegExp;
  maxFiles?: number;
  preprocessedDataPath?: string;
}): Promise<void> {
  const pipeline = new DocumentIngestionPipeline(options?.preprocessedDataPath);

  try {
    await pipeline.initialize();

    const pipelineOptions: { forceReindex?: boolean; filePattern?: RegExp; maxFiles?: number } = {};
    
    if (options?.forceReindex !== undefined) {
      pipelineOptions.forceReindex = options.forceReindex;
    }
    if (options?.filePattern !== undefined) {
      pipelineOptions.filePattern = options.filePattern;
    }
    if (options?.maxFiles !== undefined) {
      pipelineOptions.maxFiles = options.maxFiles;
    }

    const result = await pipeline.runPipeline(pipelineOptions);

    if (!result.success) {
      throw new Error("Pipeline execution failed");
    }

    console.log("[Ingestion Pipeline] Pipeline completed successfully");

  } finally {
    await pipeline.cleanup();
  }
}

/**
 * Export default class
 */
export default DocumentIngestionPipeline;
