import * as dotenv from 'dotenv';
import * as fs from 'fs/promises';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import { OpenAI } from 'openai';
import readline from 'readline';

// Load environment variables
dotenv.config();

/**
 * Environment Setup Script for SPJIMR PGPM RAG Backend
 *
 * Helps users configure their environment variables and validate setup
 */
class EnvironmentSetup {
  private rl: readline.Interface;

  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  /**
   * Prompt user for input
   */
  private async prompt(question: string): Promise<string> {
    return new Promise((resolve) => {
      this.rl.question(question, (answer) => {
        resolve(answer.trim());
      });
    });
  }

  /**
   * Display welcome message
   */
  private displayWelcome(): void {
    console.log('');
    console.log('╔══════════════════════════════════════════════════════════════╗');
    console.log('║              SPJIMR PGPM RAG Backend Setup                   ║');
    console.log('║                                                              ║');
    console.log('║  This script will help you configure your environment       ║');
    console.log('║  variables and validate your setup.                         ║');
    console.log('║                                                              ║');
    console.log('╚══════════════════════════════════════════════════════════════╝');
    console.log('');
  }

  /**
   * Generate .env file template
   */
  private getEnvTemplate(): string {
    return `# SPJIMR PGPM RAG Backend Configuration
# Copy this file to .env and fill in your actual values

# ===== REQUIRED CONFIGURATION =====

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# ===== SERVER CONFIGURATION =====

# Server Settings
NODE_ENV=development
PORT=8000
FRONTEND_URL=http://localhost:3000

# ===== INGESTION CONFIGURATION =====

# Document Ingestion Settings
RUN_INGESTION_ON_STARTUP=false
FORCE_REINDEX=false
MAX_INGESTION_FILES=0

# ===== OPTIONAL CONFIGURATION =====

# Rate Limiting (requests per 15 minutes)
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100

# Session Configuration
SESSION_TIMEOUT_MS=1800000

# Logging Level (error, warn, info, debug)
LOG_LEVEL=info

# ===== DEVELOPMENT CONFIGURATION =====

# Enable development features
ENABLE_DEBUG_ROUTES=true
ENABLE_CORS_ALL_ORIGINS=false

# ===== PRODUCTION CONFIGURATION =====
# Uncomment and configure for production deployment

# SSL/TLS Configuration
# SSL_CERT_PATH=/path/to/cert.pem
# SSL_KEY_PATH=/path/to/key.pem

# Database Connection Pool
# DB_POOL_MIN=2
# DB_POOL_MAX=10

# Monitoring
# SENTRY_DSN=https://your-sentry-dsn
# PROMETHEUS_PORT=9090
`;
  }

  /**
   * Create .env file from template
   */
  async createEnvFile(): Promise<void> {
    console.log('[Setup] Creating .env file...');

    const envPath = path.join(process.cwd(), '.env');
    const envExamplePath = path.join(process.cwd(), '.env.example');

    try {
      // Check if .env already exists
      try {
        await fs.access(envPath);
        const overwrite = await this.prompt('.env file already exists. Overwrite? (y/N): ');
        if (overwrite.toLowerCase() !== 'y') {
          console.log('[Setup] Keeping existing .env file');
          return;
        }
      } catch {
        // File doesn't exist, continue
      }

      // Create .env from template
      const template = this.getEnvTemplate();
      await fs.writeFile(envPath, template);
      await fs.writeFile(envExamplePath, template);

      console.log('[Setup] ✅ Created .env and .env.example files');
      console.log('[Setup] Please edit .env with your actual configuration values');

    } catch (error) {
      console.error('[Setup] ❌ Failed to create .env file:', error);
      throw error;
    }
  }

  /**
   * Validate environment variables
   */
  validateEnvironment(): { valid: boolean; errors: string[]; warnings: string[] } {
    console.log('[Setup] Validating environment variables...');

    const errors: string[] = [];
    const warnings: string[] = [];

    // Required variables
    const required = {
      OPENAI_API_KEY: 'OpenAI API key',
      SUPABASE_URL: 'Supabase project URL',
      SUPABASE_SERVICE_ROLE_KEY: 'Supabase service role key'
    };

    // Check required variables
    Object.entries(required).forEach(([key, description]) => {
      const value = process.env[key];
      if (!value) {
        errors.push(`Missing ${key} (${description})`);
      } else if (value.includes('your-') || value.includes('here')) {
        errors.push(`${key} appears to be a placeholder value`);
      }
    });

    // Validate specific formats
    if (process.env['SUPABASE_URL']) {
      if (!process.env['SUPABASE_URL'].startsWith('https://') || !process.env['SUPABASE_URL'].includes('.supabase.co')) {
        warnings.push('SUPABASE_URL format might be incorrect');
      }
    }

    if (process.env['OPENAI_API_KEY']) {
      if (!process.env['OPENAI_API_KEY'].startsWith('sk-')) {
        warnings.push('OPENAI_API_KEY format might be incorrect');
      }
    }

    if (process.env['PORT']) {
      const port = parseInt(process.env['PORT'], 10);
      if (isNaN(port) || port < 1 || port > 65535) {
        errors.push('PORT must be a valid port number (1-65535)');
      }
    }

    // Check optional but important variables
    if (!process.env['NODE_ENV']) {
      warnings.push('NODE_ENV not set, defaulting to development');
    }

    if (!process.env['FRONTEND_URL']) {
      warnings.push('FRONTEND_URL not set, CORS might not work correctly');
    }

    const valid = errors.length === 0;

    if (valid) {
      console.log('[Setup] ✅ Environment validation passed');
    } else {
      console.log('[Setup] ❌ Environment validation failed');
    }

    if (errors.length > 0) {
      console.log('\nErrors:');
      errors.forEach(error => console.log(`  - ${error}`));
    }

    if (warnings.length > 0) {
      console.log('\nWarnings:');
      warnings.forEach(warning => console.log(`  - ${warning}`));
    }

    return { valid, errors, warnings };
  }

  /**
   * Test OpenAI connection
   */
  async testOpenAI(): Promise<boolean> {
    console.log('[Setup] Testing OpenAI connection...');

    const apiKey = process.env['OPENAI_API_KEY'];
    if (!apiKey) {
      console.log('[Setup] ❌ OPENAI_API_KEY not set');
      return false;
    }

    try {
      const openai = new OpenAI({ apiKey });

      // Test with a simple embeddings call
      const response = await openai.embeddings.create({
        model: process.env['EMBEDDING_MODEL'] || 'text-embedding-3-small',
        input: 'test',
        encoding_format: 'float'
      });

      if (response.data && response.data.length > 0) {
        console.log('[Setup] ✅ OpenAI connection successful');
        console.log(`[Setup] Model: ${response.model}`);
        console.log(`[Setup] Embedding dimensions: ${response.data[0]?.embedding.length || 'unknown'}`);
        return true;
      } else {
        console.log('[Setup] ❌ OpenAI returned unexpected response');
        return false;
      }

    } catch (error: any) {
      console.log('[Setup] ❌ OpenAI connection failed:', error.message);

      if (error.status === 401) {
        console.log('[Setup] Hint: Check your OPENAI_API_KEY');
      } else if (error.status === 429) {
        console.log('[Setup] Hint: Rate limit exceeded, try again later');
      } else if (error.status === 404) {
        console.log('[Setup] Hint: Check your EMBEDDING_MODEL setting');
      }

      return false;
    }
  }

  /**
   * Test Supabase connection
   */
  async testSupabase(): Promise<boolean> {
    console.log('[Setup] Testing Supabase connection...');

    const url = process.env['SUPABASE_URL'];
    const key = process.env['SUPABASE_SERVICE_ROLE_KEY'];

    if (!url || !key) {
      console.log('[Setup] ❌ Supabase credentials not set');
      return false;
    }

    try {
      const supabase = createClient(url, key);

      // Test connection with a simple query
      const { data, error } = await supabase
        .from('information_schema.tables')
        .select('table_name')
        .limit(1);

      if (error) {
        console.log('[Setup] ❌ Supabase connection failed:', error.message);
        return false;
      }

      console.log('[Setup] ✅ Supabase connection successful');

      // Check if pgvector extension is available
      const { data: extensions } = await supabase
        .from('pg_available_extensions')
        .select('name')
        .eq('name', 'vector');

      if (extensions && extensions.length > 0) {
        console.log('[Setup] ✅ pgvector extension available');
      } else {
        console.log('[Setup] ⚠️ pgvector extension not found - you may need to enable it');
      }

      return true;

    } catch (error: any) {
      console.log('[Setup] ❌ Supabase connection failed:', error.message);

      if (error.message.includes('Invalid API key')) {
        console.log('[Setup] Hint: Check your SUPABASE_SERVICE_ROLE_KEY');
      } else if (error.message.includes('Invalid URL')) {
        console.log('[Setup] Hint: Check your SUPABASE_URL format');
      }

      return false;
    }
  }

  /**
   * Setup database schema
   */
  async setupDatabase(): Promise<boolean> {
    console.log('[Setup] Setting up database schema...');

    const url = process.env['SUPABASE_URL'];
    const key = process.env['SUPABASE_SERVICE_ROLE_KEY'];

    if (!url || !key) {
      console.log('[Setup] ❌ Supabase credentials not available');
      return false;
    }

    try {
      const supabase = createClient(url, key);

      // Enable pgvector extension
      console.log('[Setup] Enabling pgvector extension...');
      const { error: extError } = await supabase.rpc('exec_sql', {
        sql: 'CREATE EXTENSION IF NOT EXISTS vector;'
      });

      if (extError) {
        console.log('[Setup] ⚠️ Could not enable pgvector extension:', extError.message);
        console.log('[Setup] You may need to enable it manually in the Supabase dashboard');
      } else {
        console.log('[Setup] ✅ pgvector extension enabled');
      }

      // Create documents table
      console.log('[Setup] Creating documents table...');
      const createTableSQL = `
        CREATE TABLE IF NOT EXISTS spjimr_docs (
          id BIGSERIAL PRIMARY KEY,
          content TEXT NOT NULL,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
          embedding VECTOR(1536) NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
      `;

      const { error: tableError } = await supabase.rpc('exec_sql', {
        sql: createTableSQL
      });

      if (tableError) {
        console.log('[Setup] ❌ Could not create table:', tableError.message);
        return false;
      }

      console.log('[Setup] ✅ Documents table created/verified');

      // Create indexes
      console.log('[Setup] Creating indexes...');
      const indexSQL = `
        CREATE INDEX IF NOT EXISTS spjimr_docs_embedding_idx
        ON spjimr_docs USING hnsw (embedding vector_cosine_ops);

        CREATE INDEX IF NOT EXISTS spjimr_docs_metadata_program_idx
        ON spjimr_docs USING gin ((metadata->'program'));
      `;

      const { error: indexError } = await supabase.rpc('exec_sql', {
        sql: indexSQL
      });

      if (indexError) {
        console.log('[Setup] ⚠️ Could not create indexes:', indexError.message);
      } else {
        console.log('[Setup] ✅ Indexes created');
      }

      // Create search function
      console.log('[Setup] Creating search function...');
      const functionSQL = `
        CREATE OR REPLACE FUNCTION match_documents (
          query_embedding VECTOR(1536),
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
            1 - (spjimr_docs.embedding <=> query_embedding) AS similarity
          FROM spjimr_docs
          WHERE metadata @> filter
          ORDER BY spjimr_docs.embedding <=> query_embedding
          LIMIT match_count;
        END;
        $$;
      `;

      const { error: funcError } = await supabase.rpc('exec_sql', {
        sql: functionSQL
      });

      if (funcError) {
        console.log('[Setup] ❌ Could not create search function:', funcError.message);
        return false;
      }

      console.log('[Setup] ✅ Search function created');
      console.log('[Setup] ✅ Database schema setup complete');

      return true;

    } catch (error: any) {
      console.log('[Setup] ❌ Database setup failed:', error.message);
      return false;
    }
  }

  /**
   * Run interactive setup
   */
  async runInteractiveSetup(): Promise<void> {
    console.log('\nStarting interactive setup...\n');

    // Step 1: Create .env file
    const createEnv = await this.prompt('Create .env file from template? (Y/n): ');
    if (createEnv.toLowerCase() !== 'n') {
      await this.createEnvFile();
      console.log('\nPlease edit your .env file with the correct values, then run this script again.\n');
      return;
    }

    // Step 2: Validate environment
    const validation = this.validateEnvironment();
    if (!validation.valid) {
      console.log('\nPlease fix the environment errors and run this script again.\n');
      return;
    }

    // Step 3: Test connections
    console.log('\nTesting external service connections...\n');

    const openaiOk = await this.testOpenAI();
    const supabaseOk = await this.testSupabase();

    if (!openaiOk || !supabaseOk) {
      console.log('\nPlease fix the connection issues and run this script again.\n');
      return;
    }

    // Step 4: Setup database
    const setupDb = await this.prompt('\nSetup database schema? (Y/n): ');
    if (setupDb.toLowerCase() !== 'n') {
      await this.setupDatabase();
    }

    console.log('\n✅ Setup completed successfully!');
    console.log('\nNext steps:');
    console.log('1. Run document ingestion: npm run ingest');
    console.log('2. Start the development server: npm run dev');
    console.log('3. Test the API: curl http://localhost:8000/api/health');
  }

  /**
   * Run validation only
   */
  async runValidationOnly(): Promise<void> {
    console.log('Running validation checks...\n');

    // Validate environment
    const validation = this.validateEnvironment();

    if (validation.valid) {
      console.log('\nTesting external connections...\n');

      const openaiOk = await this.testOpenAI();
      const supabaseOk = await this.testSupabase();

      if (openaiOk && supabaseOk) {
        console.log('\n✅ All validation checks passed!');
      } else {
        console.log('\n❌ Some connection tests failed');
        process.exit(1);
      }
    } else {
      console.log('\n❌ Environment validation failed');
      process.exit(1);
    }
  }

  /**
   * Display help information
   */
  static displayHelp(): void {
    console.log(`
SPJIMR PGPM RAG Backend Environment Setup

Usage: npm run setup [options]

Options:
  --interactive    Run interactive setup wizard (default)
  --validate-only  Run validation checks only
  --create-env     Create .env file from template only
  --setup-db       Setup database schema only
  --help           Show this help message

Examples:
  npm run setup                    # Interactive setup
  npm run setup -- --validate-only # Validate current configuration
  npm run setup -- --create-env    # Create .env template only
  npm run setup -- --setup-db      # Setup database schema only

Environment Variables:
  OPENAI_API_KEY                Required - OpenAI API key
  SUPABASE_URL                  Required - Supabase project URL
  SUPABASE_SERVICE_ROLE_KEY     Required - Supabase service role key
  EMBEDDING_MODEL               Optional - Embedding model name
  NODE_ENV                      Optional - Environment (development/production)
  PORT                          Optional - Server port (default: 8000)
`);
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.rl.close();
  }

  /**
   * Run the setup script
   */
  async run(): Promise<void> {
    const args = process.argv.slice(2);

    try {
      this.displayWelcome();

      if (args.includes('--help')) {
        EnvironmentSetup.displayHelp();
        return;
      }

      if (args.includes('--create-env')) {
        await this.createEnvFile();
      } else if (args.includes('--validate-only')) {
        await this.runValidationOnly();
      } else if (args.includes('--setup-db')) {
        this.validateEnvironment();
        await this.setupDatabase();
      } else {
        // Default: interactive setup
        await this.runInteractiveSetup();
      }

    } catch (error) {
      console.error('\n❌ Setup failed:', error);
      process.exit(1);
    } finally {
      await this.cleanup();
    }
  }
}

// Run if executed directly
if (require.main === module) {
  const setup = new EnvironmentSetup();
  setup.run();
}

export default EnvironmentSetup;
