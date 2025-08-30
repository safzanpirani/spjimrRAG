import { OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { Document } from "@langchain/core/documents";
import { DocumentMetadata } from "../types/rag";
import { env } from "../config/environment";

/**
 * Vector Store Manager for SPJIMR PGPM RAG System
 *
 * Handles all vector store operations including setup, indexing, and retrieval
 * using Supabase with pgvector extension and OpenAI embeddings.
 */
export class SPJIMRVectorStore {
  private supabaseClient: SupabaseClient;
  private embeddings: OpenAIEmbeddings;
  private vectorStore: SupabaseVectorStore | null = null;
  private isInitialized: boolean = false;

  // Configuration
  private readonly tableName = "spjimr_docs";
  private readonly matchFunctionName = "match_documents";
  private readonly embeddingDimensions = 1536; // text-embedding-3-small dimensions

  constructor(
    supabaseUrl?: string,
    supabaseServiceKey?: string,
    embeddingModel?: string
  ) {
    // Initialize Supabase client
    const url = supabaseUrl || env.get('SUPABASE_URL');
    const key = supabaseServiceKey || env.get('SUPABASE_SERVICE_ROLE_KEY');

    if (!url || !key) {
      throw new Error("Supabase URL and Service Role Key are required");
    }

    this.supabaseClient = createClient(url, key);

    // Initialize embeddings
    this.embeddings = new OpenAIEmbeddings({
      modelName: embeddingModel || env.get('EMBEDDING_MODEL'),
      batchSize: 512,
      stripNewLines: true,
    });

    console.log("[Vector Store] SPJIMRVectorStore initialized");
  }

  /**
   * Initialize the vector store with database setup
   */
  async initialize(): Promise<void> {
    try {
      console.log("[Vector Store] Initializing vector store...");

      // Setup database schema
      await this.setupDatabaseSchema();

      // Initialize the LangChain vector store
      this.vectorStore = await SupabaseVectorStore.fromExistingIndex(
        this.embeddings,
        {
          client: this.supabaseClient,
          tableName: this.tableName,
          queryName: this.matchFunctionName
        }
      );

      this.isInitialized = true;
      console.log("[Vector Store] Vector store initialized successfully");

    } catch (error) {
      console.error("[Vector Store] Failed to initialize:", error);
      throw new Error(`Vector store initialization failed: ${error}`);
    }
  }

  /**
   * setting up the database schema for vector operations
   */
  private async setupDatabaseSchema(): Promise<void> {
    console.log("[Vector Store] Setting up database schema...");

    try {
      try {
        await this.supabaseClient.rpc('exec_sql', { sql: 'CREATE EXTENSION IF NOT EXISTS vector;' });
      } catch {
      }

      // Check if table already exists first
      const { data: tableExists } = await this.supabaseClient
        .from(this.tableName)
        .select('id')
        .limit(1);

      if (tableExists || tableExists === null) {
        // if table exists or query succeeded (even with no data)
        console.log("[Vector Store] Table already exists, skipping creation");
      } else {
        // try creating table using exec_sql if available
        const createTableQuery = `
          CREATE TABLE IF NOT EXISTS ${this.tableName} (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            embedding VECTOR(${this.embeddingDimensions}) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
          );
        `;

        try {
          await this.supabaseClient.rpc('exec_sql', { sql: createTableQuery });
        } catch {
          // skip if exec_sql is not present
        }
      }

      // Create indexes for better performance
      await this.createIndexes();

      // Create the similarity search function
      await this.createMatchFunction();

      console.log("[Vector Store] Database schema setup complete");
    } catch (error: any) {
      console.error("[Vector Store] Schema setup failed:", error.message);
      // Don't throw error for missing functions - continue with manual setup
      if (!error.message?.includes('exec_sql')) {
        throw error;
      }
      console.log("[Vector Store] Continuing without automated schema setup");
    }
  }

  /**
   * Create database indexes for performance
   */
  private async createIndexes(): Promise<void> {
    const indexQueries = [
      // Vector similarity index using HNSW
      `CREATE INDEX IF NOT EXISTS ${this.tableName}_embedding_idx
       ON ${this.tableName} USING hnsw (embedding vector_cosine_ops);`,

      // Metadata indexes for filtering
      `CREATE INDEX IF NOT EXISTS ${this.tableName}_metadata_program_idx
       ON ${this.tableName} USING gin ((metadata->'program'));`,

      `CREATE INDEX IF NOT EXISTS ${this.tableName}_metadata_type_idx
       ON ${this.tableName} USING gin ((metadata->'type'));`,

      // Full-text search index on content
      `CREATE INDEX IF NOT EXISTS ${this.tableName}_content_search_idx
       ON ${this.tableName} USING gin (to_tsvector('english', content));`,

      // Created at index for time-based queries
      `CREATE INDEX IF NOT EXISTS ${this.tableName}_created_at_idx
       ON ${this.tableName} (created_at DESC);`
    ];

    for (const query of indexQueries) {
      try {
        await this.supabaseClient.rpc('exec_sql', { sql: query });
      } catch {
        // Quietly skip if exec_sql is not present
      }
    }
  }

  /**
   * Create the similarity search function
   */
  private async createMatchFunction(): Promise<void> {
    const functionQuery = `
      CREATE OR REPLACE FUNCTION ${this.matchFunctionName} (
        query_embedding VECTOR(${this.embeddingDimensions}),
        match_count INT DEFAULT 5,
        filter JSONB DEFAULT '{}'::jsonb
      )
      RETURNS TABLE (
        id BIGINT,
        content TEXT,
        metadata JSONB,
        similarity FLOAT
      )
      LANGUAGE plpgsql
      AS $$
      #variable_conflict use_column
      BEGIN
        RETURN QUERY
        SELECT
          id,
          content,
          metadata,
          1 - (${this.tableName}.embedding <=> query_embedding) AS similarity
        FROM ${this.tableName}
        WHERE metadata @> filter
        ORDER BY ${this.tableName}.embedding <=> query_embedding
        LIMIT match_count;
      END;
      $$;
    `;

    try {
      await this.supabaseClient.rpc('exec_sql', { sql: functionQuery });
    } catch {
      //  skip if exec_sql is not present
      return;
    }
  }

  /**
   * Add processed documents to the vector store
   */
  async addDocuments(documents: Document[]): Promise<void> {
    if (!this.isInitialized || !this.vectorStore) {
      throw new Error("Vector store not initialized. Call initialize() first.");
    }

    if (documents.length === 0) {
      console.warn("[Vector Store] No documents to add");
      return;
    }

    console.log(`[Vector Store] Adding ${documents.length} documents...`);

    try {
      // Filter and validate documents
      const validDocuments = documents.filter(doc => {
        const content = doc.pageContent?.trim();
        return content && content.length > 20;
      });

      if (validDocuments.length === 0) {
        console.warn("[Vector Store] No valid documents to add");
        return;
      }

      // Add documents to vector store
      const ids = await this.vectorStore.addDocuments(validDocuments);

      console.log(`[Vector Store] Successfully added ${validDocuments.length} documents`);
      console.log(`[Vector Store] Generated ${ids.length} document IDs`);

    } catch (error) {
      console.error("[Vector Store] Error adding documents:", error);
      throw new Error(`Failed to add documents: ${error}`);
    }
  }

  /**
   * Perform similarity search
   */
  async similaritySearch(
    query: string,
    k: number = 5,
    filter?: Record<string, any>
  ): Promise<Document[]> {
    if (!this.isInitialized || !this.vectorStore) {
      throw new Error("Vector store not initialized. Call initialize() first.");
    }

    // Validate and sanitize query parameter
    if (!query || typeof query !== 'string') {
      throw new Error(`Invalid query parameter: expected string, got ${typeof query}. Value: ${JSON.stringify(query)}`);
    }

    const sanitizedQuery = query.toString().trim();
    if (!sanitizedQuery) {
      throw new Error("Query cannot be empty after sanitization");
    }

    console.log(`[Vector Store] Performing similarity search for: "${sanitizedQuery.substring(0, 50)}..."`);

    try {
      // Apply default filter for PGPM program
      const searchFilter = {
        program: "PGPM",
        ...filter
      };

      const results = await this.vectorStore.similaritySearch(sanitizedQuery, k, searchFilter);

      console.log(`[Vector Store] Found ${results.length} similar documents`);
      return results;

    } catch (error) {
      console.error("[Vector Store] Error in similarity search:", error);
      throw new Error(`Similarity search failed: ${error}`);
    }
  }

  /**
   * Perform similarity search with scores
   */
  async similaritySearchWithScore(
    query: string,
    k: number = 5,
    filter?: Record<string, any>
  ): Promise<[Document, number][]> {
    if (!this.isInitialized || !this.vectorStore) {
      throw new Error("Vector store not initialized. Call initialize() first.");
    }

    // Validate and sanitize query parameter
    if (!query || typeof query !== 'string') {
      throw new Error(`Invalid query parameter: expected string, got ${typeof query}. Value: ${JSON.stringify(query)}`);
    }

    const sanitizedQuery = query.toString().trim();
    if (!sanitizedQuery) {
      throw new Error("Query cannot be empty after sanitization");
    }

    try {
      const searchFilter = {
        program: "PGPM",
        ...filter
      };

      const results = await this.vectorStore.similaritySearchWithScore(sanitizedQuery, k, searchFilter);

      console.log(`[Vector Store] Found ${results.length} documents with scores`);
      return results;

    } catch (error) {
      console.error("[Vector Store] Error in similarity search with score:", error);
      throw new Error(`Similarity search with score failed: ${error}`);
    }
  }

  /**
   * Delete documents by filter
   */
  async deleteDocuments(filter: Record<string, any>): Promise<void> {
    console.log("[Vector Store] Deleting documents with filter:", filter);

    try {
      // For metadata-based filtering, use JSONB containment operator
      let query = this.supabaseClient.from(this.tableName).delete();
      
      // If filter contains fields that should be in metadata, use JSONB filtering
      if (filter['program'] || filter['type'] || filter['source']) {
        // Use metadata containment operator @>
        query = query.filter('metadata', 'cs', JSON.stringify(filter));
      } else {
        // Use direct column matching for other fields
        query = query.match(filter);
      }

      const { error } = await query;

      if (error) {
        throw new Error(`Delete failed: ${error.message}`);
      }

      console.log("[Vector Store] Documents deleted successfully");

    } catch (error) {
      console.error("[Vector Store] Error deleting documents:", error);
      throw error;
    }
  }

  /**
   * Get document count
   */
  async getDocumentCount(filter?: Record<string, any>): Promise<number> {
    try {
      let query = this.supabaseClient
        .from(this.tableName)
        .select('id', { count: 'exact', head: true });

      if (filter) {
        // Apply metadata filter
        for (const [key, value] of Object.entries(filter)) {
          query = query.eq(`metadata->${key}`, value);
        }
      }

      const { count, error } = await query;

      if (error) {
        throw new Error(`Count query failed: ${error.message}`);
      }

      return count || 0;

    } catch (error) {
      console.error("[Vector Store] Error getting document count:", error);
      return 0;
    }
  }

  /**
   * Get vector store statistics
   */
  async getStatistics(): Promise<{
    totalDocuments: number;
    documentsByType: Record<string, number>;
    lastUpdated: string | null;
  }> {
    try {
      // Get total count
      const totalDocuments = await this.getDocumentCount();

      // Get count by type
      const { data: typeData, error } = await this.supabaseClient
        .from(this.tableName)
        .select('metadata')
        .not('metadata->type', 'is', null);

      const documentsByType: Record<string, number> = {};

      if (!error && typeData) {
        typeData.forEach(row => {
          const type = row.metadata?.type || 'unknown';
          documentsByType[type] = (documentsByType[type] || 0) + 1;
        });
      }

      // Get last updated
      const { data: lastDoc } = await this.supabaseClient
        .from(this.tableName)
        .select('created_at')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      return {
        totalDocuments,
        documentsByType,
        lastUpdated: lastDoc?.created_at || null
      };

    } catch (error) {
      console.error("[Vector Store] Error getting statistics:", error);
      return {
        totalDocuments: 0,
        documentsByType: {},
        lastUpdated: null
      };
    }
  }

  /**
   * Health check for the vector store
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'unhealthy';
    details: Record<string, any>;
  }> {
    try {
      // Check database connection
      const { data, error } = await this.supabaseClient
        .from(this.tableName)
        .select('id')
        .limit(1);

      if (error) {
        return {
          status: 'unhealthy',
          details: { error: error.message }
        };
      }

      // Check embeddings
      try {
        await this.embeddings.embedQuery("test");
      } catch (embeddingError) {
        return {
          status: 'unhealthy',
          details: { embeddingError: embeddingError }
        };
      }

      const stats = await this.getStatistics();

      return {
        status: 'healthy',
        details: {
          initialized: this.isInitialized,
          totalDocuments: stats.totalDocuments,
          documentsByType: stats.documentsByType,
          lastUpdated: stats.lastUpdated
        }
      };

    } catch (error) {
      return {
        status: 'unhealthy',
        details: { error: String(error) }
      };
    }
  }

  /**
   * cleanup and close connections on shutdown
   */
  async cleanup(): Promise<void> {
    console.log("[Vector Store] Cleaning up vector store...");
    this.isInitialized = false;
    this.vectorStore = null;
  }
}

// Singleton instance for the application
let vectorStoreInstance: SPJIMRVectorStore | null = null;

/**
 * Get or create the singleton vector store instance
 */
export const getVectorStore = async (): Promise<SPJIMRVectorStore> => {
  if (!vectorStoreInstance) {
    vectorStoreInstance = new SPJIMRVectorStore();
    await vectorStoreInstance.initialize();
  }
  return vectorStoreInstance;
};

/**
 * Initialize the vector store with documents
 */
export const initializeVectorStore = async (documents: Document[]): Promise<void> => {
  console.log("[Vector Store] Initializing vector store with documents...");

  const store = await getVectorStore();

  // Check if documents already exist
  const existingCount = await store.getDocumentCount();

  if (existingCount > 0) {
    console.log(`[Vector Store] Found ${existingCount} existing documents`);
    const shouldReindex = env.get('FORCE_REINDEX');

    if (!shouldReindex) {
      console.log("[Vector Store] Skipping indexing (documents already exist)");
      return;
    }

    console.log("[Vector Store] Force reindexing enabled, clearing existing documents...");
    await store.deleteDocuments({ program: "PGPM" });
  }

  // Add new documents
  await store.addDocuments(documents);

  // Log final statistics
  const stats = await store.getStatistics();
  console.log("[Vector Store] Indexing complete:", stats);
};

/**
 * Export default configured instance
 */
export default SPJIMRVectorStore;
