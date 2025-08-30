import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { v4 as uuidv4 } from 'uuid';
import { createRAGWorkflow, testWorkflow } from './langgraph/workflow';
import { ConversationMemory } from './shared/conversationMemory';
import { buildCompositeQuery } from './shared/conversation';
import { getVectorStore } from './langchain/vectorStore';
import { DocumentIngestionPipeline } from './langchain/ingestionPipeline';
import { RAGState } from './types/rag';
import { env } from './config/environment';
import * as dotenv from 'dotenv';
import CallbackHandler from 'langfuse-langchain';
import { Langfuse } from 'langfuse';

// Load environment variables
dotenv.config();

/**
 * SPJIMR PGPM RAG Chat Server
 *
 * Express server providing RAG-powered chat API for SPJIMR PGPM information.
 * Supports both streaming and non-streaming responses with comprehensive
 * validation pipeline via LangGraph workflow.
 */
class SPJIMRChatServer {
  public app: express.Application;
  private server: any = null;
  private workflow: any = null;
  private isInitialized: boolean = false;

  // Session management for conversation memory
  private sessions: Map<string, ConversationMemory> = new Map();
  private readonly sessionTimeout = 30 * 60 * 1000; // 30 minutes

  constructor() {
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  /**
   * Setup Express middleware
   */
  private setupMiddleware(): void {
    // Security middleware
    this.app.use(helmet({
      crossOriginEmbedderPolicy: false,
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          connectSrc: ["'self'", "https://api.openai.com", env.get('SUPABASE_URL')].filter(Boolean),
        },
      },
    }));

    // CORS configuration
    this.app.use(cors({
      origin: env.get('FRONTEND_URL'),
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Session-ID']
    }));

    // Compression and parsing (disable for SSE streaming route)
    const compressionFilter = (req: Request, res: Response) => {
      try {
        // Do not compress Server-Sent Events
        if (req.path === '/api/chat/stream') {
          return false;
        }
        // @ts-ignore - using compression's default filter
        return (compression as any).filter(req, res);
      } catch {
        return true;
      }
    };
    this.app.use(compression({ filter: compressionFilter }) as any);
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use('/api/', limiter);

    // Request logging
    this.app.use((req: Request, res: Response, next: NextFunction) => {
      console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
      next();
    });

    // Session handling
    this.app.use(this.handleSession.bind(this));
  }

  /**
   * Handle session management for conversation memory
   */
  private handleSession(req: Request, res: Response, next: NextFunction): void {
    let sessionId = req.headers['x-session-id'] as string;

    if (!sessionId) {
      sessionId = uuidv4();
      res.setHeader('X-Session-ID', sessionId);
    }

    // Create or retrieve session
    if (!this.sessions.has(sessionId)) {
      this.sessions.set(sessionId, new ConversationMemory());
    }

    // Add session info to request
    (req as any).sessionId = sessionId;
    (req as any).conversationMemory = this.sessions.get(sessionId);

    // Cleanup old sessions
    this.cleanupSessions();

    next();
  }

  /**
   * Cleanup expired sessions
   */
  private cleanupSessions(): void {
    const now = Date.now();
    for (const [sessionId, memory] of this.sessions.entries()) {
      const lastActivity = (memory as any).lastActivity || now;
      if (now - lastActivity > this.sessionTimeout) {
        this.sessions.delete(sessionId);
      }
    }
  }

  /**
   * Setup API routes
   */
  private setupRoutes(): void {
    // Health check
    this.app.get('/api/health', this.handleHealthCheck.bind(this));

    // System status
    this.app.get('/api/status', this.handleSystemStatus.bind(this));

    // Chat endpoints
    this.app.post('/api/chat', this.handleChatQuery.bind(this));
    this.app.post('/api/chat/stream', this.handleStreamingChat.bind(this));

    // Session management
    this.app.get('/api/session/history', this.handleGetHistory.bind(this));
    this.app.delete('/api/session/clear', this.handleClearSession.bind(this));

    // Document management
    this.app.get('/api/documents/stats', this.handleDocumentStats.bind(this));
    this.app.post('/api/documents/reindex', this.handleReindexDocuments.bind(this));

    // Development/admin endpoints
    if (env.get('NODE_ENV') === 'development') {
      this.app.post('/api/dev/test-workflow', this.handleTestWorkflow.bind(this));
      this.app.get('/api/dev/config', this.handleGetConfig.bind(this));
    }

    // Default route
    this.app.get('/', (req: Request, res: Response) => {
      res.json({
        message: 'SPJIMR PGPM RAG Chat API',
        version: '1.0.0',
        status: 'running',
        endpoints: {
          health: '/api/health',
          chat: '/api/chat',
          stream: '/api/chat/stream'
        }
      });
    });
  }

  /**
   * Health check endpoint
   */
  private async handleHealthCheck(req: Request, res: Response): Promise<void> {
    try {
      const checks = {
        server: 'healthy',
        workflow: 'checking...',
        vectorStore: 'checking...',
        ragChain: 'removed'
      };

      // Test workflow
      try {
        if (this.workflow) {
          const testResult = await testWorkflow();
          checks.workflow = testResult ? 'healthy' : 'unhealthy';
        } else {
          checks.workflow = 'not initialized';
        }
      } catch (error) {
        checks.workflow = 'unhealthy';
      }

      // Test vector store
      try {
        const vectorStore = await getVectorStore();
        const health = await vectorStore.healthCheck();
        checks.vectorStore = health.status;
      } catch (error) {
        checks.vectorStore = 'unhealthy';
      }

      const allHealthy = Object.values(checks).every(status => status === 'healthy');

      res.status(allHealthy ? 200 : 503).json({
        status: allHealthy ? 'healthy' : 'unhealthy',
        checks,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      res.status(500).json({
        status: 'error',
        error: String(error),
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * System status endpoint with detailed information
   */
  private async handleSystemStatus(req: Request, res: Response): Promise<void> {
    try {
      const vectorStore = await getVectorStore();
      const vectorStats = await vectorStore.getStatistics();

      res.json({
        status: 'running',
        initialized: this.isInitialized,
        system: {
          nodeVersion: process.version,
          uptime: process.uptime(),
          memoryUsage: process.memoryUsage(),
        },
        vectorStore: vectorStats,
        ragConfig: { pipeline: 'workflow' },
        sessions: {
          active: this.sessions.size,
          total: this.sessions.size
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      res.status(500).json({
        status: 'error',
        error: String(error),
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Handle chat query using LangGraph workflow
   */
  private async handleChatQuery(req: Request, res: Response): Promise<void> {
    try {
      const { question, includeContext = false } = req.body;
      const sessionId = (req as any).sessionId;
      const conversationMemory = (req as any).conversationMemory as ConversationMemory;

      if (!question || typeof question !== 'string') {
        res.status(400).json({
          error: 'Question is required and must be a string',
          code: 'INVALID_QUESTION'
        });
        return;
      }

      if (!this.workflow) {
        res.status(503).json({
          error: 'RAG workflow not initialized',
          code: 'WORKFLOW_NOT_READY'
        });
        return;
      }

      console.log(`[Chat API] Processing query from session ${sessionId}: "${question.substring(0, 50)}..."`);

      // Build composite question consistently for both endpoints
      const compositeQuestion = buildCompositeQuery(question, conversationMemory);

      // Prepare initial state
      const initialState: RAGState = {
        query: compositeQuestion.trim(),
        isRelevant: false,
        retrievedDocs: [],
        hasAnswer: false,
        context: '',
        response: '',
        needsValidation: false,
        confidence: 0,
        sources: [],
        conversationHistory: []
      };

      // Run the workflow
      const result = await this.workflow.invoke(initialState);

      // Extract response details
      const response = {
        answer: result.response,
        confidence: result.confidence,
        sources: result.sources,
        isRelevant: result.isRelevant,
        sessionId
      };

      // Include context if requested (for debugging)
      if (includeContext) {
        (response as any).context = result.context;
        (response as any).retrievedDocuments = result.retrievedDocs?.length || 0;
      }

      // Add to conversation memory
      conversationMemory.addTurn({
        question,
        answer: result.response,
        confidence: result.confidence,
        sources: result.sources
      });

      // Update session activity
      (conversationMemory as any).lastActivity = Date.now();

      res.json(response);

    } catch (error) {
      console.error('[Chat API] Error processing query:', error);

      res.status(500).json({
        error: 'Failed to process query',
        code: 'PROCESSING_ERROR',
        details: env.get('NODE_ENV') === 'development' ? String(error) : undefined
      });
    }
  }

  /**
   * Handle streaming chat using Server-Sent Events
   */
  private async handleStreamingChat(req: Request, res: Response): Promise<void> {
    try {
      const { question } = req.body;
      const sessionId = (req as any).sessionId;
      const conversationMemory = (req as any).conversationMemory as ConversationMemory;

      if (!question || typeof question !== 'string') {
        res.status(400).json({
          error: 'Question is required and must be a string',
          code: 'INVALID_QUESTION'
        });
        return;
      }

      if (!this.workflow) {
        res.status(503).json({
          error: 'RAG workflow not initialized',
          code: 'WORKFLOW_NOT_READY'
        });
        return;
      }

      console.log(`[Streaming API] Processing streaming query from session ${sessionId}`);

      // Setup SSE headers
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': env.get('FRONTEND_URL'),
        'Access-Control-Allow-Credentials': 'true'
      });

      // Ensure headers are sent immediately
      try { res.flushHeaders(); } catch {}

      // Send initial connection event
      res.write(`data: ${JSON.stringify({ type: 'connected', sessionId, timestamp: new Date().toISOString() })}\n\n`);
      try { (res as any).flush?.(); } catch {}

      // Keepalive ping to prevent proxies from closing idle connections
      const ping = setInterval(() => {
        res.write(`data: ${JSON.stringify({ type: 'ping', t: Date.now() })}\n\n`);
      }, 15000);

      try {
        // Send status update
        res.write(`data: ${JSON.stringify({ type: 'status', message: 'Processing query...' })}\n\n`);
        try { (res as any).flush?.(); } catch {}

        // Build composite question consistently
        const compositeQuestion = buildCompositeQuery(question, conversationMemory);

        // Use LangGraph workflow with Node 5 disabled for enhanced retrieval + speed
        if (!this.workflow) {
          throw new Error('Workflow not initialized');
        }

        // Prepare initial state for the workflow
        const initialState: RAGState = {
          query: compositeQuestion.trim(),
          isRelevant: false,
          retrievedDocs: [],
          hasAnswer: false,
          context: '',
          response: '',
          needsValidation: false,
          confidence: 0,
          sources: [],
          conversationHistory: []
        };

        console.log("[Streaming API] Using LangGraph workflow (Node 5 disabled for speed)");
        
        // Stream tokens from Node 4 (generation) via workflow's configurable streamCallback
        let fullResponse = '';
        const streamCallback = (token: string) => {
          fullResponse += token;
          res.write(`data: ${JSON.stringify({ type: 'token', content: token })}\n\n`);
          try { (res as any).flush?.(); } catch {}
        };

        // Prepare Langfuse callback handler & trace when keys are present
        const hasLangfuse = !!(process.env['LANGFUSE_PUBLIC_KEY'] && process.env['LANGFUSE_SECRET_KEY']);
        let langfuseCallbacks: any[] = [];
        let langfuseTrace: any | undefined = undefined;
        if (hasLangfuse) {
          const lf = new Langfuse({
            publicKey: process.env['LANGFUSE_PUBLIC_KEY'] as string,
            secretKey: process.env['LANGFUSE_SECRET_KEY'] as string,
            baseUrl: (process.env['LANGFUSE_BASE_URL'] as string) || 'https://cloud.langfuse.com',
          });
          langfuseTrace = (lf as any).trace?.({ name: 'spjimr-rag', userId: sessionId, metadata: { route: 'stream', env: env.get('NODE_ENV') } });
          const handler = new CallbackHandler({ root: langfuseTrace });
          langfuseCallbacks = [handler];
        }

        // Use the sophisticated LangGraph workflow with enhanced retrieval
        // Attach trace input (now that we have compositeQuestion)
        try {
          (langfuseTrace as any)?.update?.({
            input: {
              sessionId,
              question,
              compositeQuestion,
              historyTurns: conversationMemory.getHistory().length,
              streaming: true
            }
          });
        } catch {}

        const result = await this.workflow.invoke(initialState, {
          configurable: { streamCallback, langfuseCallbacks, langfuseTrace }
        }) as RAGState;

        // Send final result
        res.write(`data: ${JSON.stringify({ type: 'complete', answer: result.response || fullResponse, confidence: result.confidence || 0.7, sources: result.sources || [], retrievedDocuments: result.retrievedDocs?.length || 0 })}\n\n`);
        try { (res as any).flush?.(); } catch {}

        // Update root trace output with final result
        try {
          (langfuseTrace as any)?.update?.({
            output: {
              answer: result.response || fullResponse,
              confidence: result.confidence || 0.7,
              sources: result.sources || [],
              retrievedDocs: result.retrievedDocs?.length || 0,
              responseLength: (result.response || fullResponse || '').length,
              processing_complete: true
            }
          });
        } catch {}

        // Add to conversation memory
        conversationMemory.addTurn({
          question,
          answer: result.response || fullResponse,
          confidence: result.confidence || 0.7,
          sources: result.sources || []
        });

        // Update session activity
        (conversationMemory as any).lastActivity = Date.now();

      } catch (error) {
        console.error('[Streaming API] Error in streaming:', error);

        res.write(`data: ${JSON.stringify({ type: 'error', error: 'Failed to process streaming query', details: env.get('NODE_ENV') === 'development' ? String(error) : undefined })}\n\n`);
        try { (res as any).flush?.(); } catch {}
      }

      // Send end event
      res.write(`data: ${JSON.stringify({ type: 'end' })}\n\n`);
      res.end();
      clearInterval(ping);

    } catch (error) {
      console.error('[Streaming API] Setup error:', error);

      if (!res.headersSent) {
        res.status(500).json({
          error: 'Failed to setup streaming',
          code: 'STREAMING_ERROR'
        });
      }
    }
  }

  /**
   * Get conversation history for a session
   */
  private handleGetHistory(req: Request, res: Response): void {
    try {
      const conversationMemory = (req as any).conversationMemory as ConversationMemory;
      const sessionId = (req as any).sessionId;

      const history = conversationMemory.getHistory();
      const stats = conversationMemory.getStats();

      res.json({
        sessionId,
        history,
        stats,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      res.status(500).json({
        error: 'Failed to retrieve history',
        details: String(error)
      });
    }
  }

  /**
   * Clear conversation history for a session
   */
  private handleClearSession(req: Request, res: Response): void {
    try {
      const conversationMemory = (req as any).conversationMemory as ConversationMemory;
      const sessionId = (req as any).sessionId;

      conversationMemory.clear();

      res.json({
        message: 'Session cleared successfully',
        sessionId,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      res.status(500).json({
        error: 'Failed to clear session',
        details: String(error)
      });
    }
  }

  /**
   * Get document statistics
   */
  private async handleDocumentStats(req: Request, res: Response): Promise<void> {
    try {
      const vectorStore = await getVectorStore();
      const stats = await vectorStore.getStatistics();

      res.json({
        ...stats,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      res.status(500).json({
        error: 'Failed to retrieve document statistics',
        details: String(error)
      });
    }
  }

  /**
   * Reindex documents (admin endpoint)
   */
  private async handleReindexDocuments(req: Request, res: Response): Promise<void> {
    try {
      const { force = false } = req.body;

      console.log('[Reindex API] Starting document reindexing...');

      const pipeline = new DocumentIngestionPipeline();
      await pipeline.initialize();

      const result = await pipeline.runPipeline({
        forceReindex: force
      });

      res.json({
        message: 'Reindexing completed',
        result,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('[Reindex API] Error:', error);

      res.status(500).json({
        error: 'Reindexing failed',
        details: String(error),
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Test workflow (development only)
   */
  private async handleTestWorkflow(req: Request, res: Response): Promise<void> {
    try {
      const { query = "What is SPJIMR PGPM?" } = req.body;

      const result = await testWorkflow(query);

      res.json({
        message: 'Workflow test completed',
        query,
        result,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      res.status(500).json({
        error: 'Workflow test failed',
        details: String(error)
      });
    }
  }

  /**
   * Get configuration (development only)
   */
  private async handleGetConfig(req: Request, res: Response): Promise<void> {
    try {
      res.json({
        environment: env.get('NODE_ENV'),
        config: {
          retrieval: { pipeline: 'workflow' },
          generation: { model: 'gpt-4o-mini' },
          embedding: {
            model: env.get('EMBEDDING_MODEL')
          }
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      res.status(500).json({
        error: 'Failed to retrieve configuration',
        details: String(error)
      });
    }
  }

  /**
   * Setup error handling middleware
   */
  private setupErrorHandling(): void {
    // 404 handler
    this.app.use((req: Request, res: Response) => {
      res.status(404).json({
        error: 'Endpoint not found',
        path: req.path,
        method: req.method
      });
    });

    // Global error handler
    this.app.use((error: any, req: Request, res: Response, next: NextFunction) => {
      console.error('[Server Error]:', error);

      // Don't send error details in production
      const isDevelopment = env.get('NODE_ENV') === 'development';

      res.status(error.status || 500).json({
        error: 'Internal server error',
        code: error.code || 'INTERNAL_ERROR',
        details: isDevelopment ? String(error) : undefined,
        timestamp: new Date().toISOString()
      });
    });
  }

  /**
   * Initialize the server and all components
   */
  async initialize(): Promise<void> {
    try {
      console.log('[Server] Initializing SPJIMR PGPM RAG Chat Server...');

      // Validate environment variables
      this.validateEnvironment();

      // Initialize RAG workflow
      console.log('[Server] Creating RAG workflow...');
      this.workflow = createRAGWorkflow();

      // Initialize vector store
      console.log('[Server] Initializing vector store...');
      await getVectorStore();

      // Test the workflow
      console.log('[Server] Testing workflow...');
      const testResult = await testWorkflow();
      if (!testResult) {
        console.warn('[Server] Workflow test failed, but continuing...');
      }

      this.isInitialized = true;
      console.log('[Server] Server initialization completed successfully');

    } catch (error) {
      console.error('[Server] Initialization failed:', error);
      throw new Error(`Server initialization failed: ${error}`);
    }
  }

  /**
   * Validate required environment variables
   */
  private validateEnvironment(): void {
    // Environment validation is now handled by the Environment class
    console.log('[Server] Environment variables validated');
  }

  /**
   * Start the server
   */
  async start(port: number = 8000): Promise<void> {
    try {
      // Initialize if not already done
      if (!this.isInitialized) {
        await this.initialize();
      }

      // Start the server
      this.server = this.app.listen(port, () => {
        console.log(`[Server] SPJIMR PGPM RAG Chat Server running on port ${port}`);
        console.log(`[Server] Environment: ${env.get('NODE_ENV')}`);
        console.log(`[Server] Health check: http://localhost:${port}/api/health`);
        console.log(`[Server] API docs: http://localhost:${port}/`);
      });

      // Handle server errors
      this.server.on('error', (error: any) => {
        if (error.code === 'EADDRINUSE') {
          console.error(`[Server] Port ${port} is already in use`);
        } else {
          console.error('[Server] Server error:', error);
        }
        process.exit(1);
      });

      // Graceful shutdown
      this.setupGracefulShutdown();

    } catch (error) {
      console.error('[Server] Failed to start server:', error);
      throw error;
    }
  }

  /**
   * Setup graceful shutdown handlers
   */
  private setupGracefulShutdown(): void {
    const shutdown = async (signal: string) => {
      console.log(`[Server] Received ${signal}, shutting down gracefully...`);

      if (this.server) {
        this.server.close(async () => {
          console.log('[Server] HTTP server closed');

          try {
            // Cleanup resources
            const vectorStore = await getVectorStore();
            await vectorStore.cleanup();

            console.log('[Server] Cleanup completed');
          } catch (error) {
            console.error('[Server] Error during cleanup:', error);
          }

          process.exit(0);
        });
      } else {
        process.exit(0);
      }
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
  }
}

// Create and export server instance
const server = new SPJIMRChatServer();

export default server;
export { SPJIMRChatServer };
