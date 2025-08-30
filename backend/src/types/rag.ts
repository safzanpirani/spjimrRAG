import { Document } from "@langchain/core/documents";

// Main RAG State for LangGraph workflow
export interface RAGState {
  query: string;
  isRelevant: boolean;
  retrievedDocs: Document[];
  hasAnswer: boolean;
  context: string;
  response: string;
  needsValidation: boolean;
  confidence?: number;
  sources?: string[];
  errorMessage?: string;
  conversationHistory?: ChatMessage[];
}

// Valid query categories for SPJIMR PGPM
export const VALID_QUERY_CATEGORIES = [
  'eligibility',      // Work experience, marks, degree requirements
  'admissions',       // Process, deadlines, requirements
  'curriculum',       // Courses, majors, minors, structure
  'fees',            // Fee structure, payment options
  'placements',      // Statistics, companies, salary
  'duration',        // Program length, phases
  'campus',          // Location, facilities
  'accreditation',   // AACSB, AMBA, EQUIS
  'rankings',        // FT, India rankings
  'faculty',         // Teaching staff info
  'followup',        // Follow-up queries in conversation
  'general',         // General/broad information requests
] as const;

export type QueryCategory = typeof VALID_QUERY_CATEGORIES[number];

// Response templates for different scenarios
export const RESPONSE_TEMPLATES = {
  NOT_RELEVANT: "I can only answer questions about the SPJIMR PGPM program. Please ask about admissions, curriculum, fees, placements, or eligibility.",

  NO_INFORMATION: "I don't have enough information to answer this specific question about the PGPM program. Please contact SPJIMR directly for more details.",

  PARTIAL_INFO: "Based on the available information: {partial_answer}. For complete details, please check with SPJIMR admissions.",

  VERIFICATION_FAILED: "I'm having trouble processing this request right now. Please try rephrasing your question or contact SPJIMR directly."
} as const;

// Document metadata structure
export interface DocumentMetadata {
  source: string;
  program: "PGPM";
  type: "admissions" | "curriculum" | "fees" | "placements" | "eligibility" | "general";
  page?: number;
  section?: string;
  lastUpdated?: string;
  sourceFile?: string;
  similarityScore?: number;
}

// Query validation result
export interface QueryValidationResult {
  isRelevant: boolean;
  category: QueryCategory | undefined;
  reason: string;
  confidence: number;
}

// Context validation result
export interface ContextValidationResult {
  hasAnswer: boolean;
  confidence: number;
  reasoning: string | undefined;
}

// Response validation result
export interface ResponseValidationResult {
  isValid: boolean;
  issues: string[];
  confidence: number;
  reasoning: string | undefined;
}

// Node execution result
export interface NodeResult<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: Record<string, any>;
}

// Chat message types
export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  metadata?: {
    sources?: string[];
    confidence?: number;
    category?: QueryCategory;
  };
}

// RAG configuration
export interface RAGConfig {
  chunkSize: number;
  chunkOverlap: number;
  maxRetrievalDocs: number;
  embeddingModel: string;
  llmModel: string;
  temperature: number;
  confidenceThreshold: number;
}

// Error types
export class RAGError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message);
    this.name = "RAGError";
  }
}

export class ValidationError extends RAGError {
  constructor(message: string, details?: any) {
    super(message, "VALIDATION_ERROR", details);
    this.name = "ValidationError";
  }
}

export class RetrievalError extends RAGError {
  constructor(message: string, details?: any) {
    super(message, "RETRIEVAL_ERROR", details);
    this.name = "RetrievalError";
  }
}

export class GenerationError extends RAGError {
  constructor(message: string, details?: any) {
    super(message, "GENERATION_ERROR", details);
    this.name = "GenerationError";
  }
}
