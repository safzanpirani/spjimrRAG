import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import {
  RAGState,
  QueryValidationResult,
  VALID_QUERY_CATEGORIES,
  ValidationError,
  NodeResult
} from "../../types/rag";
import { getActiveLangfuseTrace, getActiveLangfuseCallbacks } from "../../shared/tracing";
import { hasRelevantKeywords, isKnownPGPMTerm, suggestCategories } from "../../shared/queryHeuristics";

// Validation response schema
const QueryValidationSchema = z.object({
  isRelevant: z.boolean(),
  category: z.enum(VALID_QUERY_CATEGORIES).optional(),
  reason: z.string(),
  confidence: z.number().min(0).max(1)
});

// Query validation prompt template
const QUERY_VALIDATION_PROMPT = ChatPromptTemplate.fromTemplate(`
You are a query classifier for the SPJIMR PGPM (Post Graduate Programme in Management) program.

Your task is to determine if the user's query is directly related to the SPJIMR PGPM program OR is a follow-up query in an ongoing conversation about SPJIMR PGPM.

CONTEXT: You will see the full conversation context including previous Q&A pairs. Pay attention to this context.

VALID QUERIES (answer with isRelevant: true):
1. Direct SPJIMR PGPM topics:
   - Eligibility criteria, admissions process, curriculum, fees, placements
   - Program duration, campus info, accreditation, rankings, faculty
   - Statistics, data, comprehensive information requests

2. PGPM-specific terms and initiatives:
   - "abhyudaya", "docc", "sitaras", "samavesh", "intex" (PGPM programs/initiatives)
   - "aicte", "aacsb", "amba" (accreditation bodies)
   - "ppt", "cis" (PGPM academic terms)
   - Partner institutions: "mccombs", "insead", "cornell", "michigan", "barcelona", "reutlingen"

3. Follow-up queries in SPJIMR PGPM context:
   - "longer", "more details", "elaborate", "tell me more"
   - "all the stats", "give me everything", "complete information"
   - "what else", "any other", "additional details"
   - Clarification requests after previous SPJIMR PGPM responses

4. Contextual references:
   - Queries that only make sense in context of previous SPJIMR PGPM discussion
   - Pronoun references ("it", "this", "that") referring to previously discussed PGPM topics

INVALID TOPICS (answer with isRelevant: false):
- Completely unrelated topics with no conversation context
- Other MBA programs when not comparing to SPJIMR PGPM
- General business advice unrelated to the conversation

IMPORTANT: If the conversation context shows previous discussion about SPJIMR PGPM, then follow-up queries like "longer", "more", "stats" should be considered RELEVANT.

Full Query Context: {query}

Analyze the entire context and respond with JSON in this exact format:
{{
  "isRelevant": boolean,
  "category": "eligibility|admissions|curriculum|fees|placements|duration|campus|accreditation|rankings|faculty|followup|general" (use "followup" for continuation queries, "general" for broad information requests),
  "reason": "brief explanation considering conversation context",
  "confidence": number between 0 and 1
}}
`);

/**
 * Query Validation Node
 *
 * This is the first node in the RAG workflow that determines if the user's query
 * is relevant to the SPJIMR PGPM program. It acts as a gatekeeper to ensure
 * we only process queries that are within our knowledge domain.
 */
export const queryValidationNode = async (state: RAGState): Promise<Partial<RAGState>> => {
  try {
    const root = getActiveLangfuseTrace();
    const step = root?.span?.({ name: "validate_query" }) || root?.span?.("validate_query");
    try { step?.update?.({ input: { query: state.query } }); } catch {}
    console.log(`[Query Validation] Processing query: "${state.query}"`);

    // Initialize the LLM with strict settings
    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini", // Using gpt-4o-mini as specified in plan
      temperature: 0, // Zero temperature for consistent classification
      maxTokens: 500,
      callbacks: getActiveLangfuseCallbacks() || [],
    });

    // Create the validation chain
    const chain = QUERY_VALIDATION_PROMPT.pipe(llm);

    // Execute the validation
    const result = await chain.invoke({ query: state.query });

    // Parse the JSON response
    let validationResult: QueryValidationResult;
    try {
      const parsedResult = JSON.parse(result.content as string);
      validationResult = {
        isRelevant: parsedResult.isRelevant,
        category: parsedResult.category || undefined,
        reason: parsedResult.reason,
        confidence: parsedResult.confidence
      };
    } catch (parseError) {
      console.error("[Query Validation] Failed to parse LLM response:", parseError);
      throw new ValidationError("Failed to parse validation response", {
        rawResponse: result.content,
        parseError: parseError
      });
    }

    console.log(`[Query Validation] Result:`, {
      isRelevant: validationResult.isRelevant,
      category: validationResult.category,
      confidence: validationResult.confidence,
      reason: validationResult.reason
    });

    // Handle extremely short or vague but obviously in-domain prompts
    const fallbackRelevant = hasRelevantKeywords(state.query) || 
      /\b(pgpm|spjimr|mba)\b/i.test(state.query) || 
      state.query.trim().length <= 3 ||
      isKnownPGPMTerm(state.query);

    // Return updated state
    const updateState: any = {
      isRelevant: validationResult.isRelevant || fallbackRelevant,
      confidence: validationResult.confidence,
    };
    
    if (validationResult.isRelevant) {
      updateState.sources = [];
    }
    
    try {
      step?.update?.({
        output: {
          isRelevant: updateState.isRelevant,
          category: validationResult.category,
          confidence: validationResult.confidence,
          reason: validationResult.reason
        }
      });
    } catch {}
    step?.end?.();
    return updateState;

  } catch (error) {
    console.error("[Query Validation] Error:", error);
    try { (getActiveLangfuseTrace() as any)?.span?.({ name: "validate_query_error" })?.end?.(); } catch {}

    if (error instanceof ValidationError) {
      throw error;
    }

    throw new ValidationError("Query validation failed", {
      query: state.query,
      originalError: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Helper function to check if a query contains PGMP-related keywords
 * This can be used as a fallback or pre-filter before LLM validation
 */
// moved to shared/queryHeuristics

/**
 * Check if a query is a known PGPM-specific term that should be considered relevant
 */
// moved to shared/queryHeuristics

/**
 * Get relevant categories based on query content
 * This is a simple keyword-based approach that can complement LLM classification
 */
// moved to shared/queryHeuristics
