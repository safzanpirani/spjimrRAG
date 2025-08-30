import * as dotenv from 'dotenv';
import { DocumentIngestionPipeline } from '../langchain/ingestionPipeline';
import { getVectorStore } from '../langchain/vectorStore';

// Load environment variables
dotenv.config();

/**
 * Document Ingestion Script for SPJIMR PGPM RAG System
 *
 * Standalone script to ingest and index PDF documents from the preprocessed-data directory.
 * Can be run independently of the main server for batch processing.
 */
class DocumentIngestionScript {
  private pipeline: DocumentIngestionPipeline;
  private readonly options: {
    forceReindex: boolean;
    maxFiles: number;
    filePattern?: RegExp;
    preprocessedDataPath?: string;
    validate: boolean;
    dryRun: boolean;
  };

  constructor(options: {
    forceReindex?: boolean;
    maxFiles?: number;
    filePattern?: string;
    preprocessedDataPath?: string;
    validate?: boolean;
    dryRun?: boolean;
  } = {}) {
    const pipelineOptions: any = {
      forceReindex: options.forceReindex || false,
      maxFiles: options.maxFiles || 0,
      validate: options.validate || false,
      dryRun: options.dryRun || false
    };
    
    if (options.filePattern) {
      pipelineOptions.filePattern = new RegExp(options.filePattern, 'i');
    }
    if (options.preprocessedDataPath) {
      pipelineOptions.preprocessedDataPath = options.preprocessedDataPath;
    }
    
    this.options = pipelineOptions;

    this.pipeline = new DocumentIngestionPipeline(this.options.preprocessedDataPath);

    console.log('[Ingestion Script] Document Ingestion Script initialized');
    console.log('[Ingestion Script] Options:', {
      forceReindex: this.options.forceReindex,
      maxFiles: this.options.maxFiles || 'unlimited',
      filePattern: options.filePattern || 'none',
      validate: this.options.validate,
      dryRun: this.options.dryRun
    });
  }

  /**
   * Display help information
   */
  static displayHelp(): void {
    console.log(`
SPJIMR PGPM Document Ingestion Script

Usage: npm run ingest [options]

Options:
  --force-reindex     Force reindexing of all documents (clears existing data)
  --max-files <n>     Limit processing to first n files
  --file-pattern <p>  Process only files matching regex pattern
  --data-path <path>  Custom path to preprocessed data directory
  --validate          Run validation checks after ingestion
  --dry-run          Show what would be processed without actually doing it
  --help             Show this help message

Examples:
  npm run ingest                           # Normal ingestion (skip if data exists)
  npm run ingest -- --force-reindex        # Force reindex all documents
  npm run ingest -- --max-files 5          # Process only first 5 files
  npm run ingest -- --file-pattern ".*handbook.*"  # Process only handbook files
  npm run ingest -- --validate             # Run with validation
  npm run ingest -- --dry-run             # Preview what would be processed

Environment Variables:
  OPENAI_API_KEY                Required - OpenAI API key for embeddings
  SUPABASE_URL                  Required - Supabase project URL
  SUPABASE_SERVICE_ROLE_KEY     Required - Supabase service role key
  EMBEDDING_MODEL               Optional - Embedding model (default: text-embedding-3-small)
  FORCE_REINDEX                 Optional - Force reindex on startup
  MAX_INGESTION_FILES           Optional - Maximum files to process
`);
  }

  /**
   * Validate environment setup
   */
  private validateEnvironment(): void {
    console.log('[Ingestion Script] Validating environment...');

    const required = [
      'OPENAI_API_KEY',
      'SUPABASE_URL',
      'SUPABASE_SERVICE_ROLE_KEY'
    ];

    const missing = required.filter(key => !process.env[key]);

    if (missing.length > 0) {
      console.error('[Ingestion Script] Missing required environment variables:');
      missing.forEach(key => console.error(`  - ${key}`));
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }

    console.log('[Ingestion Script] Environment validation passed');
  }

  /**
   * Display current system status
   */
  private async displaySystemStatus(): Promise<void> {
    console.log('[Ingestion Script] Checking system status...');

    try {
      // Check vector store
      const vectorStore = await getVectorStore();
      const health = await vectorStore.healthCheck();
      const stats = await vectorStore.getStatistics();

      console.log('[Ingestion Script] System Status:');
      console.log(`  Vector Store: ${health.status}`);
      console.log(`  Existing Documents: ${stats.totalDocuments}`);

      if (stats.documentsByType && Object.keys(stats.documentsByType).length > 0) {
        console.log('  Document Types:');
        Object.entries(stats.documentsByType).forEach(([type, count]) => {
          console.log(`    - ${type}: ${count}`);
        });
      }

      console.log(`  Last Updated: ${stats.lastUpdated || 'Never'}`);

    } catch (error) {
      console.warn('[Ingestion Script] Could not get system status:', error);
    }
  }

  /**
   * Run dry run to show what would be processed
   */
  private async runDryRun(): Promise<void> {
    console.log('[Ingestion Script] Running dry run (no actual processing)...');

    try {
      await this.pipeline.initialize();

      // Discover files that would be processed
      const allFiles = await this.pipeline.discoverMarkdownFiles();
      let filesToProcess = allFiles;

      // Apply file pattern filter
      if (this.options.filePattern) {
        filesToProcess = allFiles.filter(file => this.options.filePattern!.test(file));
        console.log(`[Ingestion Script] Files matching pattern: ${filesToProcess.length}/${allFiles.length}`);
      }

      // Apply max files limit
      if (this.options.maxFiles > 0) {
        filesToProcess = filesToProcess.slice(0, this.options.maxFiles);
        console.log(`[Ingestion Script] Limited to first ${filesToProcess.length} files`);
      }

      console.log(`[Ingestion Script] Would process ${filesToProcess.length} files:`);

      filesToProcess.forEach((file, index) => {
        const fileName = file.split(/[/\\]/).pop() || file;
        console.log(`  ${index + 1}. ${fileName}`);
      });

      if (filesToProcess.length === 0) {
        console.log('[Ingestion Script] No files would be processed with current filters');
      }

    } catch (error) {
      console.error('[Ingestion Script] Dry run failed:', error);
      throw error;
    } finally {
      await this.pipeline.cleanup();
    }
  }

  /**
   * Run the actual ingestion process
   */
  private async runIngestion(): Promise<void> {
    console.log('[Ingestion Script] Starting document ingestion...');

    try {
      await this.pipeline.initialize();

      // Show progress updates
      const progressInterval = setInterval(() => {
        const progress = this.pipeline.getProgress();
        if (progress.totalFiles > 0) {
          console.log(`[Ingestion Script] Progress: ${progress.progressPercentage}% (${progress.processedFiles}/${progress.totalFiles} files, ${progress.indexedDocuments} documents indexed)`);
        }
      }, 10000); // Update every 10 seconds

      // Run the pipeline
      const pipelineRunOptions: any = {};
      
      if (this.options.forceReindex !== undefined) {
        pipelineRunOptions.forceReindex = this.options.forceReindex;
      }
      if (this.options.filePattern !== undefined) {
        pipelineRunOptions.filePattern = this.options.filePattern;
      }
      if (this.options.maxFiles > 0) {
        pipelineRunOptions.maxFiles = this.options.maxFiles;
      }
      
      const result = await this.pipeline.runPipeline(pipelineRunOptions);

      // Clear progress interval
      clearInterval(progressInterval);

      // Display results
      console.log('\n[Ingestion Script] Ingestion Results:');
      console.log(`  Success: ${result.success}`);
      console.log(`  Files Processed: ${result.processedFiles}/${result.totalFiles}`);
      console.log(`  Documents Indexed: ${result.indexedDocuments}`);
      console.log(`  Failed Files: ${result.failedFiles.length}`);
      console.log(`  Duration: ${(result.duration / 1000).toFixed(1)}s`);

      if (result.failedFiles.length > 0) {
        console.log('\n[Ingestion Script] Failed Files:');
        result.failedFiles.forEach(file => {
          const fileName = file.split(/[/\\]/).pop() || file;
          console.log(`  - ${fileName}`);
        });
      }

      if (!result.success) {
        throw new Error('Ingestion pipeline failed');
      }

    } catch (error) {
      console.error('[Ingestion Script] Ingestion failed:', error);
      throw error;
    } finally {
      await this.pipeline.cleanup();
    }
  }

  /**
   * Run validation checks
   */
  private async runValidation(): Promise<void> {
    console.log('[Ingestion Script] Running validation checks...');

    try {
      await this.pipeline.initialize();

      const validation = await this.pipeline.validateIndexedData();

      console.log('\n[Ingestion Script] Validation Results:');
      console.log(`  Valid: ${validation.valid}`);
      console.log(`  Total Documents: ${validation.statistics.totalDocuments || 0}`);

      if (validation.statistics.documentsByType) {
        console.log('  Document Types:');
        Object.entries(validation.statistics.documentsByType).forEach(([type, count]) => {
          console.log(`    - ${type}: ${count}`);
        });
      }

      if (validation.issues.length > 0) {
        console.log('\n[Ingestion Script] Validation Issues:');
        validation.issues.forEach(issue => {
          console.log(`  - ${issue}`);
        });
      }

      if (!validation.valid) {
        console.warn('[Ingestion Script] Validation failed - there may be issues with the indexed data');
      } else {
        console.log('[Ingestion Script] Validation passed - data appears to be correct');
      }

    } catch (error) {
      console.error('[Ingestion Script] Validation failed:', error);
      throw error;
    } finally {
      await this.pipeline.cleanup();
    }
  }

  /**
   * Run the complete ingestion script
   */
  async run(): Promise<void> {
    const startTime = Date.now();

    try {
      console.log('[Ingestion Script] SPJIMR PGPM Document Ingestion Script Starting...');

      // Validate environment
      this.validateEnvironment();

      // Display system status
      await this.displaySystemStatus();

      if (this.options.dryRun) {
        // Run dry run only
        await this.runDryRun();
      } else {
        // Run actual ingestion
        await this.runIngestion();

        // Run validation if requested
        if (this.options.validate) {
          await this.runValidation();
        }
      }

      const duration = (Date.now() - startTime) / 1000;
      console.log(`\n[Ingestion Script] Script completed successfully in ${duration.toFixed(1)}s`);

    } catch (error) {
      const duration = (Date.now() - startTime) / 1000;
      console.error(`[Ingestion Script] Script failed after ${duration.toFixed(1)}s:`, error);
      process.exit(1);
    }
  }
}

/**
 * Parse command line arguments
 */
function parseArgs(): {
  forceReindex?: boolean;
  maxFiles?: number;
  filePattern?: string;
  preprocessedDataPath?: string;
  validate?: boolean;
  dryRun?: boolean;
  help?: boolean;
} {
  const args = process.argv.slice(2);
  const options: any = {};

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case '--force-reindex':
        options.forceReindex = true;
        break;
      case '--max-files':
        const maxFilesArg = args[++i];
        if (!maxFilesArg) {
          throw new Error('--max-files requires a number argument');
        }
        options.maxFiles = parseInt(maxFilesArg, 10);
        if (isNaN(options.maxFiles) || options.maxFiles < 1) {
          throw new Error('--max-files must be a positive integer');
        }
        break;
      case '--file-pattern':
        const patternArg = args[++i];
        if (!patternArg) {
          throw new Error('--file-pattern requires a pattern argument');
        }
        options.filePattern = patternArg;
        break;
      case '--data-path':
        const pathArg = args[++i];
        if (!pathArg) {
          throw new Error('--data-path requires a path argument');
        }
        options.preprocessedDataPath = pathArg;
        break;
      case '--validate':
        options.validate = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--help':
        options.help = true;
        break;
      default:
        if (arg && arg.startsWith('--')) {
          throw new Error(`Unknown option: ${arg}`);
        }
    }
  }

  return options;
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  try {
    // Parse command line arguments
    const options = parseArgs();

    // Show help if requested
    if (options.help) {
      DocumentIngestionScript.displayHelp();
      return;
    }

    // Create and run the script
    const script = new DocumentIngestionScript(options);
    await script.run();

  } catch (error) {
    if (error instanceof Error && error.message.includes('Unknown option')) {
      console.error(`Error: ${error.message}`);
      console.error('Use --help for usage information');
      process.exit(1);
    }

    console.error('[Main] Script error:', error);
    process.exit(1);
  }
}

// Run if this file is executed directly
if (require.main === module) {
  main();
}

export default DocumentIngestionScript;
export { DocumentIngestionScript };
