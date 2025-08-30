import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { getActiveLangfuseCallbacks, getActiveLangfuseTrace } from "../../shared/tracing";
import {
  RAGState,
  RESPONSE_TEMPLATES,
  GenerationError
} from "../../types/rag";

// Internal streaming callback configured by the workflow wrapper
let genStreamCallback: ((token: string) => void) | undefined;
export const setActiveStreamCallback = (cb?: (token: string) => void) => {
  genStreamCallback = cb;
};

// Response generation schema
const ResponseGenerationSchema = z.object({
  response: z.string(),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().optional()
});

// Response generation prompt template for grounded responses
const GROUNDED_RESPONSE_PROMPT = ChatPromptTemplate.fromTemplate(`
You are a helpful assistant for SPJIMR PGPM program inquiries. You must follow these STRICT RULES:

CRITICAL RULES:
1. ONLY use information from the provided context
2. Do NOT make up or infer information not explicitly stated
3. If specific details are missing, clearly state so
4. Always be factual and accurate
5. Do not provide advice or opinions, only factual information
6. For follow-up queries like "longer", "more details", "stats", provide comprehensive responses
7. For statistical queries, extract and present ALL numerical data found in context
8. Always mention that users should verify information with SPJIMR directly for the most current details

QUERY TYPE HANDLING:
- If user asks for "longer", "more details", "elaborate": Provide a comprehensive, detailed response
- If user asks for "stats", "statistics", "all data": Focus on extracting ALL numerical data and statistics
- If user asks broad questions like "everything", "tell me about PGPM": Provide comprehensive overview
- For follow-up queries: Assume they want detailed information based on available context

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide comprehensive supporting details from the context
- For statistical queries: Present data in organized format (salary figures, percentages, rankings, etc.)
- End with a verification note

Context Information:
{context}

User Question: {query}

Based ONLY on the information provided in the context above, provide a helpful and accurate answer. For follow-up queries or requests for more information, be comprehensive and detailed. If any specific details are missing, explicitly mention what information is missing.

Remember: It's better to say "this information is not available in the provided context" than to make assumptions.
`);

// Fallback response prompt for edge cases
const FALLBACK_RESPONSE_PROMPT = ChatPromptTemplate.fromTemplate(`
You are a helpful assistant for SPJIMR PGPM program inquiries. 

The user asked: {query}
We cannot provide a complete answer because: {reason}

Provide a brief, natural response that:
1. Acknowledges their question about PGPM
2. Explains we don't have that specific information
3. Suggests contacting SPJIMR for details
4. Keep it conversational and under 2 sentences
`);

/**
 * Response Generation Node
 *
 * This node generates the final response to the user's query. It handles three scenarios:
 * 1. Query not relevant to SPJIMR PGPM
 * 2. No sufficient information available
 * 3. Generate grounded response from available context
 */
export const generationNode = async (state: RAGState): Promise<Partial<RAGState>> => {
  try {
    const root = getActiveLangfuseTrace();
    const step = root?.span?.({ name: "generate" }) || root?.span?.("generate");
    try { step?.update?.({ input: { query: state.query, contextChars: state.context?.length || 0, docs: (state.retrievedDocs||[]).length } }); } catch {}
    console.log(`[Generation] Generating response for query: "${state.query}"`);
    console.log(`[Generation] State - isRelevant: ${state.isRelevant}, hasAnswer: ${state.hasAnswer}`);

    // If a streaming callback is registered, enable true token streaming
    const enableStreaming = typeof genStreamCallback === "function";
    const llm = new ChatOpenAI({
      modelName: "gpt-4o-mini", // Using gpt-4o-mini as specified in plan
      temperature: 0.1, // Low temperature for consistent, factual responses
      maxTokens: 800,
      streaming: enableStreaming,
      callbacks: [
        ...(getActiveLangfuseCallbacks() || []),
        ...(enableStreaming
        ? [
            {
              handleLLMNewToken: async (token: string) => {
                try {
                  genStreamCallback?.(token);
                } catch (err) {
                  console.error("[Generation] Stream callback error:", err);
                }
              },
            } as any,
          ]
        : []),
      ],
    });

    // Case 1: Query not relevant to SPJIMR PGPM
    if (!state.isRelevant) {
      console.log("[Generation] Query not relevant, using template response");
      return {
        response: RESPONSE_TEMPLATES.NOT_RELEVANT,
        needsValidation: false,
        confidence: 1.0
      };
    }

    // Case 2: No sufficient information available (try intent-specific salvage)
    if (!state.hasAnswer || !state.context || state.context.trim().length === 0) {
      console.log("[Generation] No sufficient information available");
      console.log(`[Generation] Debug - hasAnswer: ${state.hasAnswer}, context length: ${state.context?.length || 0}`);
      console.log(`[Generation] Debug - retrievedDocs count: ${state.retrievedDocs?.length || 0}`);
      
      // For fee queries, check if we actually have fee docs even if hasAnswer is false
      const queryLower = state.query.toLowerCase();
      if ((queryLower.includes('fee') || queryLower.includes('cost')) && state.retrievedDocs && state.retrievedDocs.length > 0) {
        console.log("[Generation] Fee query detected with retrieved docs, checking for fee information");
        
        // Check if any retrieved doc contains fee information
        const feeInfo = state.retrievedDocs.find(doc => {
          const content = doc.pageContent.toLowerCase();
          return content.includes('fee') || content.includes('cost') || content.includes('rs.') || content.includes('lakh') || content.includes('rupees');
        });
        
        if (feeInfo) {
          console.log("[Generation] Found fee information in retrieved docs, generating response despite hasAnswer=false");
          const contextText = state.retrievedDocs
            .map((doc, index) => `Document ${index + 1}:\n${doc.pageContent}`)
            .join("\n---\n");
            
          const chain = GROUNDED_RESPONSE_PROMPT.pipe(llm);
          const genRes = await chain.invoke({
            context: contextText,
            query: state.query
          });

          return {
            response: genRes.content as string,
            needsValidation: true,
            confidence: 0.7
          };
        }
      }

      // Handle follow-up queries and statistical requests even when hasAnswer is false
      const isFollowUp = /^(longer|more|elaborate|tell me more|expand|details?|what else|any other|additional)$/i.test(state.query.trim());
      const isStatsQuery = /\b(stats?|statistics|data|numbers|figures|all data|give me all|show me all|tell me all)\b/i.test(state.query);
      const isBroadQuery = /^(everything|tell me everything|all information|complete|comprehensive|full)$/i.test(state.query.trim());
      
      if ((isFollowUp || isStatsQuery || isBroadQuery) && state.retrievedDocs && state.retrievedDocs.length > 0) {
        console.log("[Generation] Follow-up/stats/broad query detected with retrieved docs, generating comprehensive response");
        
        const contextText = state.retrievedDocs
          .map((doc, index) => `Document ${index + 1}:\n${doc.pageContent}`)
          .join("\n---\n");
          
        const chain = GROUNDED_RESPONSE_PROMPT.pipe(llm);
        const genRes = await chain.invoke({
          context: contextText,
          query: state.query
        });

        return {
          response: genRes.content as string,
          needsValidation: true,
          confidence: 0.8
        };
      }

      // Intent-specific salvage for duration and social projects before fallback
      const wantsDuration = /\b(duration|how long|months?|length)\b/i.test(state.query);
      const wantsSocial = /\b(abhyudaya|social\s*work|social\s*projects|docc|underprivileged|community)\b/i.test(state.query);
      
      if (wantsDuration) {
        // If any retrieved doc mentions months, synthesize a concise answer
        const monthMatch = (state.retrievedDocs || []).map((d: any) => d.pageContent.match(/(1[58])\s*-?month/gi)).flat().filter(Boolean)[0];
        if (monthMatch) {
          const months = /18/.test(monthMatch[0]!) ? '18 months' : '15 months';
          return {
            response: `The SPJIMR PGPM is ${months}. It includes 2 months online, 12 months on-campus, and 4 months for social/start-up projects and international immersion. Please verify with SPJIMR for the latest details.`,
            needsValidation: false,
            confidence: /18/.test(monthMatch[0]!) ? 0.9 : 0.6
          };
        }
      }
      
      // Salvage for social/community work queries
      if (wantsSocial && state.retrievedDocs && state.retrievedDocs.length > 0) {
        console.log("[Generation] Social query detected, checking for relevant content in retrieved docs");
        
        // Check if any doc contains social project information
        const socialDoc = state.retrievedDocs.find((doc: any) => {
          const content = doc.pageContent.toLowerCase();
          return content.includes('abhyudaya') || content.includes('docc') || 
                 content.includes('social') || content.includes('community') ||
                 content.includes('underprivileged') || content.includes('citizenship');
        });
        
        if (socialDoc) {
          console.log("[Generation] Found social project information, generating response despite hasAnswer=false");
          const contextText = state.retrievedDocs
            .map((doc: any, index: number) => `Document ${index + 1}:\n${doc.pageContent}`)
            .join("\n---\n");
            
          const chain = GROUNDED_RESPONSE_PROMPT.pipe(llm);
          const genRes = await chain.invoke({
            context: contextText,
            query: state.query
          });

          return {
            response: genRes.content as string,
            needsValidation: true,
            confidence: 0.75
          };
        }
      }

      // Generate a more contextual "don't know" response
      try {
        const fallbackChain = FALLBACK_RESPONSE_PROMPT.pipe(llm);
        const fallbackResult = await fallbackChain.invoke({
          query: state.query,
          reason: "Insufficient information in knowledge base"
        });

        return {
          response: fallbackResult.content as string,
          needsValidation: false,
          confidence: 0.3
        };
      } catch (fallbackError) {
        console.error("[Generation] Fallback response generation failed:", fallbackError);
        return {
          response: RESPONSE_TEMPLATES.NO_INFORMATION,
          needsValidation: false,
          confidence: 0.2
        };
      }
    }

    // Case 3: Generate grounded response
    console.log("[Generation] Generating grounded response from context");
    console.log(`[Generation] Context length: ${state.context.length} characters`);

    const chain = GROUNDED_RESPONSE_PROMPT.pipe(llm);
    const genRes = await chain.invoke({
      context: state.context,
      query: state.query
    });

    const response = genRes.content as string;

    // Basic quality checks on the response
    if (!response || response.trim().length < 20) {
      throw new GenerationError("Generated response is too short or empty", {
        responseLength: response?.length || 0,
        query: state.query
      });
    }

    // Check if response seems to contain hallucinated information
    if (containsProblematicContent(response)) {
      console.warn("[Generation] Response may contain problematic content, flagging for validation");
    }

    console.log(`[Generation] Generated response length: ${response.length} characters`);

    const finalRes = {
      response: response,
      needsValidation: true,
      confidence: state.confidence || 0.8
    };
    try { step?.update?.({ output: { responseLength: response.length, needsValidation: true, confidence: state.confidence || 0.8 } }); } catch {}
    step?.end?.();
    return finalRes;

  } catch (error) {
    console.error("[Generation] Error during response generation:", error);
    try { (getActiveLangfuseTrace() as any)?.span?.({ name: "generate_error" })?.end?.(); } catch {}

    if (error instanceof GenerationError) {
      throw error;
    }

    // Fallback to safe template response on error
    return {
      response: RESPONSE_TEMPLATES.VERIFICATION_FAILED,
      needsValidation: false,
      confidence: 0.1,
      errorMessage: error instanceof Error ? error.message : String(error)
    };
  }
};

/**
 * Helper function to check for potentially problematic content
 */
export const containsProblematicContent = (response: string): boolean => {
  const problematicPhrases = [
    "i think", "i believe", "probably", "might be", "could be",
    "in my opinion", "i assume", "i guess", "perhaps",
    "it seems like", "maybe", "i would say"
  ];

  const responseLower = response.toLowerCase();
  return problematicPhrases.some(phrase => responseLower.includes(phrase));
};

/**
 * Helper function to enhance response with proper citations
 */
export const enhanceResponseWithCitations = (response: string, sources: string[]): string => {
  if (!sources || sources.length === 0) {
    return response;
  }

  const citationText = "\n\nSources: " + sources.join(", ");
  return response + citationText;
};

/**
 * Helper function to add verification disclaimer
 */
export const addVerificationDisclaimer = (response: string): string => {
  const disclaimer = "\n\nPlease verify this information directly with SPJIMR admissions for the most current and accurate details.";

  if (response.toLowerCase().includes("verify") || response.toLowerCase().includes("contact")) {
    return response; // Already has verification guidance
  }

  return response + disclaimer;
};

/**
 * Helper function to truncate response if too long
 */
export const truncateResponse = (response: string, maxLength: number = 1000): string => {
  if (response.length <= maxLength) {
    return response;
  }

  // Find the last complete sentence within the limit
  const truncated = response.substring(0, maxLength);
  const lastSentenceEnd = Math.max(
    truncated.lastIndexOf('.'),
    truncated.lastIndexOf('!'),
    truncated.lastIndexOf('?')
  );

  if (lastSentenceEnd > maxLength * 0.7) {
    return truncated.substring(0, lastSentenceEnd + 1);
  }

  // If no good sentence break, just truncate with ellipsis
  return truncated + "...";
};

/**
 * Helper function to format numerical information consistently
 */
export const formatNumericalInfo = (response: string): string => {
  // Format percentages, currencies, etc. consistently
  return response
    .replace(/(\d+)%/g, "$1%") // Ensure no space before %
    .replace(/Rs\.?\s?(\d+)/g, "Rs. $1") // Standardize currency format
    .replace(/(\d{1,3})(?=(\d{3})+(?!\d))/g, "$1,"); // Add commas to large numbers
};

/**
 * Helper function to validate response quality
 */
export const validateResponseQuality = (response: string, query: string): {
  isValid: boolean;
  issues: string[];
  score: number;
} => {
  const issues: string[] = [];
  let score = 1.0;

  // Check minimum length
  if (response.length < 50) {
    issues.push("Response too short");
    score -= 0.3;
  }

  // Check for problematic content
  if (containsProblematicContent(response)) {
    issues.push("Contains uncertain language");
    score -= 0.2;
  }

  // Check if response addresses the query
  const queryWords = query.toLowerCase().split(' ').filter(w => w.length > 3);
  const responseWords = response.toLowerCase().split(' ');
  const matchedWords = queryWords.filter(word =>
    responseWords.some(rWord => rWord.includes(word))
  );

  if (matchedWords.length < queryWords.length * 0.3) {
    issues.push("Response may not address the query adequately");
    score -= 0.2;
  }

  // Check for generic responses
  if (response.includes("I don't have") && response.length < 100) {
    issues.push("Response appears too generic");
    score -= 0.1;
  }

  return {
    isValid: issues.length === 0 && score >= 0.6,
    issues,
    score: Math.max(score, 0)
  };
};
