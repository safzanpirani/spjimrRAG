import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import {
  RAGState,
  ContextValidationResult,
  ValidationError
} from "../../types/rag";
import { getAdaptiveConfidenceThreshold } from "./retrieval";
import { getActiveLangfuseTrace, getActiveLangfuseCallbacks } from "../../shared/tracing";

// Context validation response schema
const ContextValidationSchema = z.object({
  hasAnswer: z.boolean(),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().optional()
});


const CONTEXT_VALIDATION_PROMPT = ChatPromptTemplate.fromTemplate(`
You are a document analyzer for the SPJIMR PGPM program. Your task is to determine if the provided documents contain sufficient information to answer the user's question.

VALIDATION CRITERIA:
- Return hasAnswer: true if the documents contain relevant information that can answer the question
- The information should be present in the documents, but can include related details that address the question
- For eligibility questions, if the documents contain eligibility requirements, return true
- For program questions, if the documents contain program-related information, return true
- Only return false if the documents are completely unrelated to the question
- It's acceptable to have partial information as long as it's relevant and helpful

FLEXIBILITY GUIDELINES:
- Information doesn't need to be word-for-word matching the question
- Related concepts and requirements can be used to answer questions
- If documents contain admission requirements for eligibility questions, that's sufficient
- If documents contain program details for program questions, that's sufficient

Documents:
{documents}

User Question: {query}

Analyze the documents carefully and respond with JSON in this exact format:
{{
  "hasAnswer": boolean,
  "confidence": number between 0 and 1 (1 = completely certain, 0 = no confidence),
  "reasoning": "brief explanation of why you can/cannot answer based on the documents"
}}

Remember: The goal is to be helpful while staying grounded in the provided documents.
`);

/**
 * Context Validation Node
 *
 * This node validates whether the retrieved documents actually contain
 * sufficient information to answer the user's query. It acts as a quality
 * gate to prevent hallucination and ensure we only generate responses
 * when we have credible source material.
 */
export const contextCheckNode = async (state: RAGState): Promise<Partial<RAGState>> => {
  try {
    const root = getActiveLangfuseTrace();
    const step = root?.span?.({ name: "check_context" }) || root?.span?.("check_context");
    try { step?.update?.({ input: { docs: (state.retrievedDocs || []).slice(0,3).map(d=>({source: d.metadata['source'], len: d.pageContent.length})) } }); } catch {}
    console.log(`[Context Check] Validating context for query: "${state.query}"`);
    console.log(`[Context Check] Number of retrieved documents: ${state.retrievedDocs?.length || 0}`);

    // If no documents were retrieved, we clearly don't have an answer
    if (!state.retrievedDocs || state.retrievedDocs.length === 0) {
      console.log("[Context Check] No documents retrieved, cannot answer");
      return {
        hasAnswer: false,
        context: "",
        confidence: 0
      };
    }

    // Combine all document content
    const docsContent = state.retrievedDocs
      .map((doc, index) => `Document ${index + 1}:\n${doc.pageContent}`)
      .join("\n---\n");

    console.log(`[Context Check] Total context length: ${docsContent.length} characters`);

    // Initialize the LLM with strict settings
    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini", // Using gpt-4o-mini as specified in plan
      temperature: 0, // Zero temperature for consistent validation
      maxTokens: 300,
      callbacks: getActiveLangfuseCallbacks() || [],
    });

    // Create the validation chain
    const chain = CONTEXT_VALIDATION_PROMPT.pipe(llm);

    // Execute the context validation
    const ctxRes = await chain.invoke({
      documents: docsContent,
      query: state.query
    });

    // Parse the JSON response
    let validationResult: ContextValidationResult;
    try {
      const parsedResult = JSON.parse(ctxRes.content as string);
      validationResult = {
        hasAnswer: parsedResult.hasAnswer,
        confidence: parsedResult.confidence,
        reasoning: parsedResult.reasoning || undefined
      };
    } catch (parseError) {
      console.error("[Context Check] Failed to parse LLM response:", parseError);
      throw new ValidationError("Failed to parse context validation response", {
        rawResponse: ctxRes.content,
        parseError: parseError
      });
    }

    console.log(`[Context Check] Validation result:`, {
      hasAnswer: validationResult.hasAnswer,
      confidence: validationResult.confidence,
      reasoning: validationResult.reasoning
    });

    // Use adaptive confidence threshold based on query type and context quality
    const confidenceThreshold = getAdaptiveConfidenceThreshold(state.query, docsContent);
    console.log(`[Context Check] Using adaptive confidence threshold: ${confidenceThreshold} for query type`);
    
    const finalHasAnswer = validationResult.hasAnswer && validationResult.confidence >= confidenceThreshold;

    if (validationResult.hasAnswer && !finalHasAnswer) {
      console.log(`[Context Check] Confidence ${validationResult.confidence} below threshold ${confidenceThreshold}, marking as no answer`);
    }

    const out: Partial<RAGState> = {
      hasAnswer: finalHasAnswer,
      context: finalHasAnswer ? docsContent : "",
      confidence: validationResult.confidence
    };
    try {
      step?.update?.({
        output: {
          hasAnswer: finalHasAnswer,
          confidence: validationResult.confidence,
          reason: validationResult.reasoning
        }
      });
    } catch {}
    step?.end?.();
    return out;

  } catch (error) {
    console.error("[Context Check] Error during context validation:", error);
    try { (getActiveLangfuseTrace() as any)?.span?.({ name: "check_context_error" })?.end?.(); } catch {}

    if (error instanceof ValidationError) {
      throw error;
    }

    throw new ValidationError("Context validation failed", {
      query: state.query,
      documentsCount: state.retrievedDocs?.length || 0,
      originalError: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Helper function to calculate context quality score
 * This can be used to assess the overall quality of retrieved documents
 */
export const calculateContextQuality = (state: RAGState): number => {
  if (!state.retrievedDocs || state.retrievedDocs.length === 0) {
    return 0;
  }

  let qualityScore = 0;
  const queryTerms = state.query.toLowerCase().split(' ').filter(term => term.length > 2);

  for (const doc of state.retrievedDocs) {
    const content = doc.pageContent.toLowerCase();
    let docScore = 0;

    // Check for query term matches
    for (const term of queryTerms) {
      if (content.includes(term)) {
        docScore += 0.1;
      }
    }

    // Prefer longer, more detailed documents
    if (doc.pageContent.length > 500) {
      docScore += 0.1;
    }

    // Prefer documents with structured metadata
    if (doc.metadata['section'] || doc.metadata['type']) {
      docScore += 0.05;
    }

    qualityScore += Math.min(docScore, 0.25); // Cap individual document contribution
  }

  return Math.min(qualityScore, 1.0); // Cap total score at 1.0
};

/**
 * Helper function to identify potential gaps in context
 */
export const identifyContextGaps = (query: string, documents: string): string[] => {
  const gaps: string[] = [];
  const queryLower = query.toLowerCase();

  // Check for common information gaps
  if (queryLower.includes('fee') || queryLower.includes('cost')) {
    if (!documents.toLowerCase().includes('fee') && !documents.toLowerCase().includes('cost')) {
      gaps.push("No fee information found in documents");
    }
  }

  if (queryLower.includes('deadline') || queryLower.includes('date')) {
    const hasDatePattern = /\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{1,2}-\d{1,2}-\d{4}\b|\b\d{4}-\d{2}-\d{2}\b/.test(documents);
    if (!hasDatePattern) {
      gaps.push("No specific dates found in documents");
    }
  }

  if (queryLower.includes('eligibility') || queryLower.includes('requirement')) {
    if (!documents.toLowerCase().includes('eligib') && !documents.toLowerCase().includes('require')) {
      gaps.push("No eligibility requirements found in documents");
    }
  }

  if (queryLower.includes('salary') || queryLower.includes('placement')) {
    if (!documents.toLowerCase().includes('salary') && !documents.toLowerCase().includes('placement')) {
      gaps.push("No placement or salary information found in documents");
    }
  }

  return gaps;
};

/**
 * Helper function to estimate answer completeness
 */
export const estimateAnswerCompleteness = (query: string, context: string): number => {
  const queryWords = query.toLowerCase().split(' ').filter(word => word.length > 2);
  const contextLower = context.toLowerCase();

  let matchedWords = 0;
  for (const word of queryWords) {
    if (contextLower.includes(word)) {
      matchedWords++;
    }
  }

  const completeness = queryWords.length > 0 ? matchedWords / queryWords.length : 0;
  return Math.min(completeness, 1.0);
};
