/**
 * Environment Configuration
 * Centralized environment variable handling with proper typing
 */

// Load environment variables before any processing
import * as dotenv from 'dotenv';
dotenv.config();

interface EnvironmentConfig {
  // Required
  OPENAI_API_KEY: string;
  SUPABASE_URL: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  
  // Optional with defaults
  SUPABASE_ANON_KEY: string | undefined;
  PORT: number;
  NODE_ENV: 'development' | 'production' | 'test';
  FRONTEND_URL: string;
  
  // RAG Configuration
  EMBEDDING_MODEL: string;
  MAX_RETRIEVAL_DOCS: number;
  FORCE_REINDEX: boolean;
  
  // Optional startup configuration
  RUN_INGESTION_ON_STARTUP: boolean;
  MAX_INGESTION_FILES: number | undefined;
  
  // Optional LangChain
  LANGCHAIN_TRACING_V2: boolean;
  LANGCHAIN_API_KEY: string | undefined;
}

class Environment {
  private static instance: Environment;
  private config: EnvironmentConfig;

  private constructor() {
    this.config = this.loadConfig();
  }

  public static getInstance(): Environment {
    if (!Environment.instance) {
      Environment.instance = new Environment();
    }
    return Environment.instance;
  }

  private loadConfig(): EnvironmentConfig {
    // Required environment variables
    const requiredVars = {
      OPENAI_API_KEY: process.env['OPENAI_API_KEY'],
      SUPABASE_URL: process.env['SUPABASE_URL'],
      SUPABASE_SERVICE_ROLE_KEY: process.env['SUPABASE_SERVICE_ROLE_KEY']
    };

    // Check for missing required variables
    const missing = Object.entries(requiredVars)
      .filter(([_, value]) => !value)
      .map(([key]) => key);

    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }

    return {
      // Required
      OPENAI_API_KEY: requiredVars.OPENAI_API_KEY!,
      SUPABASE_URL: requiredVars.SUPABASE_URL!,
      SUPABASE_SERVICE_ROLE_KEY: requiredVars.SUPABASE_SERVICE_ROLE_KEY!,
      
      // Optional with defaults
      SUPABASE_ANON_KEY: process.env['SUPABASE_ANON_KEY'],
      PORT: parseInt(process.env['PORT'] || '8000', 10),
      NODE_ENV: (process.env['NODE_ENV'] as any) || 'development',
      FRONTEND_URL: process.env['FRONTEND_URL'] || 'http://localhost:3000',
      
      // RAG Configuration
      EMBEDDING_MODEL: process.env['EMBEDDING_MODEL'] || 'text-embedding-3-small',
      MAX_RETRIEVAL_DOCS: parseInt(process.env['MAX_RETRIEVAL_DOCS'] || '5', 10),
      FORCE_REINDEX: process.env['FORCE_REINDEX'] === 'true',
      
      // Startup configuration
      RUN_INGESTION_ON_STARTUP: process.env['RUN_INGESTION_ON_STARTUP'] === 'true',
      MAX_INGESTION_FILES: process.env['MAX_INGESTION_FILES'] ? parseInt(process.env['MAX_INGESTION_FILES'], 10) : undefined,
      
      // LangChain
      LANGCHAIN_TRACING_V2: process.env['LANGCHAIN_TRACING_V2'] === 'true',
      LANGCHAIN_API_KEY: process.env['LANGCHAIN_API_KEY']
    };
  }

  public get<K extends keyof EnvironmentConfig>(key: K): EnvironmentConfig[K] {
    return this.config[key];
  }

  public getAll(): EnvironmentConfig {
    return { ...this.config };
  }

  public isDevelopment(): boolean {
    return this.config.NODE_ENV === 'development';
  }

  public isProduction(): boolean {
    return this.config.NODE_ENV === 'production';
  }
}

// Export singleton instance
export const env = Environment.getInstance();
export type { EnvironmentConfig };