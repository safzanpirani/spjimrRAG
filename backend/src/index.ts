import * as dotenv from 'dotenv';
import path from 'path';
import { SPJIMRChatServer } from './server';
import { DocumentIngestionPipeline } from './langchain/ingestionPipeline';
import { env } from './config/environment';

// Load environment variables
dotenv.config();

/**
 * Main entry point for SPJIMR PGPM RAG Chat Backend
 *
 * Handles server initialization, document ingestion, and startup
 */
class Application {
  private server: SPJIMRChatServer;
  private readonly port: number;
  private readonly environment: string;

  constructor() {
    this.server = new SPJIMRChatServer();
    this.port = env.get('PORT');
    this.environment = env.get('NODE_ENV');

    console.log(`[Application] Starting SPJIMR PGPM RAG Chat Backend`);
    console.log(`[Application] Environment: ${this.environment}`);
    console.log(`[Application] Port: ${this.port}`);
  }

  /**
   * Initialize application components
   */
  async initialize(): Promise<void> {
    try {
      console.log('[Application] Initializing application...');

      // Check and run document ingestion if needed
      await this.checkAndRunIngestion();

      // Initialize the server
      console.log('[Application] Initializing server...');
      await this.server.initialize();

      console.log('[Application] Application initialized successfully');

    } catch (error) {
      console.error('[Application] Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Check if document ingestion is needed and run if necessary
   */
  private async checkAndRunIngestion(): Promise<void> {
    try {
      const shouldRunIngestion = env.get('RUN_INGESTION_ON_STARTUP');
      const forceReindex = env.get('FORCE_REINDEX');

      if (!shouldRunIngestion && !forceReindex) {
        console.log('[Application] Skipping document ingestion (not configured to run on startup)');
        return;
      }

      console.log('[Application] Checking document ingestion...');

      const pipeline = new DocumentIngestionPipeline();
      await pipeline.initialize();

      // Check if documents exist unless force reindex
      if (!forceReindex) {
        const stats = await pipeline.validateIndexedData();

        if (stats.valid && stats.statistics.totalDocuments > 0) {
          console.log(`[Application] Found ${stats.statistics.totalDocuments} existing documents, skipping ingestion`);
          return;
        }

        if (!stats.valid) {
          console.warn('[Application] Document validation issues found:', stats.issues);
        }
      }

      // Run document ingestion
      console.log('[Application] Running document ingestion pipeline...');

      const maxFiles = env.get('MAX_INGESTION_FILES');
      const pipelineOptions: { forceReindex: boolean; maxFiles?: number } = { forceReindex };
      if (maxFiles !== undefined) {
        pipelineOptions.maxFiles = maxFiles;
      }
      
      const result = await pipeline.runPipeline(pipelineOptions);

      if (result.success) {
        console.log('[Application] Document ingestion completed successfully');
        console.log(`[Application] Processed ${result.processedFiles}/${result.totalFiles} files`);
        console.log(`[Application] Indexed ${result.indexedDocuments} documents`);
      } else {
        console.error('[Application] Document ingestion failed');
        if (result.failedFiles.length > 0) {
          console.error('[Application] Failed files:', result.failedFiles);
        }

        // Continue startup even if ingestion fails in development
        if (this.environment === 'production') {
          throw new Error('Document ingestion failed in production environment');
        }
      }

      await pipeline.cleanup();

    } catch (error) {
      console.error('[Application] Document ingestion error:', error);

      // In production, fail hard if ingestion fails
      if (this.environment === 'production') {
        throw error;
      }

      // In development, warn but continue
      console.warn('[Application] Continuing startup despite ingestion error (development mode)');
    }
  }

  /**
   * Start the application
   */
  async start(): Promise<void> {
    try {
      // Initialize components
      await this.initialize();

      // Start the server
      console.log(`[Application] Starting server on port ${this.port}...`);
      await this.server.start(this.port);

    } catch (error) {
      console.error('[Application] Failed to start application:', error);
      process.exit(1);
    }
  }

  /**
   * Setup global error handlers
   */
  private setupErrorHandlers(): void {
    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason, promise) => {
      console.error('[Application] Unhandled Promise Rejection:', reason);
      console.error('Promise:', promise);

      // Exit gracefully in production
      if (this.environment === 'production') {
        console.error('[Application] Exiting due to unhandled promise rejection');
        process.exit(1);
      }
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      console.error('[Application] Uncaught Exception:', error);

      // Always exit on uncaught exceptions
      console.error('[Application] Exiting due to uncaught exception');
      process.exit(1);
    });

    // Handle warnings
    process.on('warning', (warning) => {
      console.warn('[Application] Warning:', warning.name, warning.message);
      if (warning.stack) {
        console.warn(warning.stack);
      }
    });
  }

  /**
   * Validate environment and configuration
   */
  private validateEnvironment(): void {
    console.log('[Application] Validating environment...');

    // Required environment variables
    const required = [
      'OPENAI_API_KEY',
      'SUPABASE_URL',
      'SUPABASE_SERVICE_ROLE_KEY'
    ];

    // Environment validation is now handled by the Environment class
    console.log('[Application] Environment variables validated via centralized config');

    // Validate Supabase URL format
    const supabaseUrl = env.get('SUPABASE_URL');
    if (!supabaseUrl.startsWith('https://') || !supabaseUrl.includes('.supabase.co')) {
      console.warn('[Application] Supabase URL format might be incorrect');
    }

    // Validate port
    if (isNaN(this.port) || this.port < 1 || this.port > 65535) {
      throw new Error(`Invalid port number: ${env.get('PORT')}`);
    }

    console.log('[Application] Environment validation completed');
  }

  /**
   * Display startup banner
   */
  private displayBanner(): void {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════╗');
    console.log('║                 SPJIMR PGPM RAG Chat Backend                 ║');
    console.log('║                                                              ║');
    console.log('║  Intelligent chat interface for SPJIMR PGPM program         ║');
    console.log('║  Powered by LangGraph + LangChain + OpenAI + Supabase       ║');
    console.log('║                                                              ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');
    console.log('');
  }

  /**
   * Run the application
   */
  async run(): Promise<void> {
    try {
      // Display banner
      this.displayBanner();

      // Setup error handlers
      this.setupErrorHandlers();

      // Validate environment
      this.validateEnvironment();

      // Start the application
      await this.start();

    } catch (error) {
      console.error('[Application] Application startup failed:', error);
      process.exit(1);
    }
  }
}

/**
 * Create and run the application
 */
async function main(): Promise<void> {
  const app = new Application();
  await app.run();
}

// Run the application if this file is executed directly
if (require.main === module) {
  main().catch(error => {
    console.error('[Main] Fatal error:', error);
    process.exit(1);
  });
}

export default main;
export { Application };
