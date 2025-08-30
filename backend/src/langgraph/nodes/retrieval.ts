import { OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { Document } from "@langchain/core/documents";
import {
  RAGState,
  DocumentMetadata,
  RetrievalError,
  RAGConfig
} from "../../types/rag";
import { enhanceQuery } from "../../shared/queryHeuristics";
import { getActiveLangfuseTrace } from "../../shared/tracing";

/**
 * Document Retrieval Node
 *
 * This node performs semantic search to retrieve relevant documents from the
 * Supabase vector store. It uses OpenAI embeddings to convert the query into
 * a vector representation and finds the most similar document chunks.
 */

// Default retrieval configuration
const DEFAULT_RETRIEVAL_CONFIG = {
  k: 5, // Number of documents to retrieve
  scoreThreshold: 0.7, // Minimum similarity score
  maxTokens: 4000, // Maximum total tokens from retrieved docs
};

export const retrievalNode = async (state: RAGState): Promise<Partial<RAGState>> => {
  try {
    const root = getActiveLangfuseTrace();
    const step = root?.span?.({ name: "retrieve" }) || root?.span?.("retrieve");
    try { step?.update?.({ input: { query: state.query } }); } catch {}
    console.log(`[Retrieval] Starting document retrieval for query: "${state.query}"`);

    // Enhanced query expansion system for all types of queries
    let searchQuery = await enhanceQuery(state.query);
    
    if (searchQuery !== state.query) {
      console.log(`[Retrieval] Enhanced query from "${state.query}" to "${searchQuery}"`);
    }

    // Initialize embeddings model
    const embeddings = new OpenAIEmbeddings({
      modelName: process.env['EMBEDDING_MODEL'] || "text-embedding-3-small",
      batchSize: 1, // Process one query at a time
      stripNewLines: true,
    });

    // Initialize Supabase client
    const supabaseUrl = process.env['SUPABASE_URL'];
    const supabaseServiceKey = process.env['SUPABASE_SERVICE_ROLE_KEY'];

    if (!supabaseUrl || !supabaseServiceKey) {
      throw new RetrievalError("Missing Supabase configuration", {
        hasUrl: !!supabaseUrl,
        hasServiceKey: !!supabaseServiceKey
      });
    }

    const supabaseClient = createClient(supabaseUrl, supabaseServiceKey);

    // Initialize vector store
    const vectorStore = await SupabaseVectorStore.fromExistingIndex(
      embeddings,
      {
        client: supabaseClient,
        tableName: "spjimr_docs",
        queryName: "match_documents"
        // Note: filter will be applied at retriever level
      }
    );

    // Perform similarity search with adaptive filter
    const maxDocs = parseInt(process.env['MAX_RETRIEVAL_DOCS'] || "8");
    const filter = createRetrievalFilter(searchQuery);
    const retriever = vectorStore.asRetriever({
      k: maxDocs,
      searchType: "similarity",
      filter
    });

    console.log(`[Retrieval] Searching for top ${maxDocs} relevant documents...`);

    let retrievedDocs = await retriever.getRelevantDocuments(searchQuery);

    // Targeted second-pass for ambiguous queries or specific intents
    const wantsDuration = /\b(duration|how long|months?|length)\b/i.test(state.query);
    const wantsSocial = /\b(abhyudaya|social\s*work|underprivileged|docc|development.*corporate.*citizenship)\b/i.test(state.query);
    const isAmbiguous = /\b(overview|everything|about|info|information)\b/i.test(state.query) || state.query.trim().split(/\s+/).length <= 2;

    // Special handling for single specific terms that are mentioned in documents
    const isSpecificTerm = /^(abhyudaya|docc|sitaras)$/i.test(state.query.trim());

    if ((wantsDuration || wantsSocial || isAmbiguous || isSpecificTerm) && retrievedDocs.length < 4) {
      console.log(`[Retrieval] Performing enhanced search for ${isSpecificTerm ? 'specific term' : 'targeted content'}`);
      
      const booster = wantsDuration
        ? `${state.query} 18-month 18 month 15-month 15 month program duration length months`
        : wantsSocial || isSpecificTerm
          ? `${state.query} social projects Abhyudaya DoCC Development Corporate Citizenship underprivileged community social impact social work NGO`
          : `SPJIMR PGPM overview curriculum admissions placements international immersion`;
      
      const extraDocs = await retriever.getRelevantDocuments(booster);
      console.log(`[Retrieval] Enhanced search found ${extraDocs.length} additional documents`);
      
      const seen = new Set(retrievedDocs.map(d => d.pageContent.slice(0, 200)));
      for (const d of extraDocs) {
        const key = d.pageContent.slice(0, 200);
        if (!seen.has(key)) {
          retrievedDocs.push(d);
          seen.add(key);
        }
      }
    }
    
    // Additional keyword-based search for very specific terms if still insufficient
    if (isSpecificTerm && retrievedDocs.length < 2) {
      console.log(`[Retrieval] Performing direct keyword search for specific term: ${state.query}`);
      const keywordDocs = await performDirectKeywordSearch(vectorStore, state.query.trim());
      
      const seen = new Set(retrievedDocs.map(d => d.pageContent.slice(0, 200)));
      for (const d of keywordDocs) {
        const key = d.pageContent.slice(0, 200);
        if (!seen.has(key)) {
          retrievedDocs.push(d);
          seen.add(key);
          console.log(`[Retrieval] Added document from keyword search: ${d.metadata['source']}`);
        }
      }
    }

    // Filter documents by relevance score if available
    // Deduplicate and filter
    const filteredDocs = deduplicateDocuments(retrievedDocs).filter((doc, index) => {
      // Log document info for debugging
      console.log(`[Retrieval] Doc ${index + 1}:`, {
        source: doc.metadata['source'],
        type: doc.metadata['type'],
        contentPreview: doc.pageContent.substring(0, 100) + "...",
        contentLength: doc.pageContent.length
      });

      // Basic content length filter
      return doc.pageContent.trim().length > 50;
    });

    // Calculate total token estimate (rough approximation: 4 chars = 1 token)
    const totalTokens = filteredDocs.reduce((sum, doc) =>
      sum + Math.ceil(doc.pageContent.length / 4), 0);

    console.log(`[Retrieval] Retrieved ${filteredDocs.length} documents`, {
      totalDocuments: retrievedDocs.length,
      filteredDocuments: filteredDocs.length,
      estimatedTokens: totalTokens
    });

    // Reorder by simple relevance heuristic
    const orderedDocs = reorderDocumentsByRelevance(filteredDocs as any, state.query);

    // Extract sources for citation
    const sources = orderedDocs
      .map(doc => doc.metadata['source'] as string)
      .filter((source, index, arr) => arr.indexOf(source) === index) // Remove duplicates
      .slice(0, 3); // Limit to top 3 sources

    const result = {
      retrievedDocs: orderedDocs,
      sources: sources
    };
    try {
      step?.update?.({
        output: {
          k: orderedDocs.length,
          sources,
          preview: orderedDocs.slice(0, 3).map(d => ({
            source: d.metadata['source'],
            type: d.metadata['type'],
            content: d.pageContent.substring(0, 200)
          }))
        }
      });
    } catch {}
    step?.end?.();
    return result;

  } catch (error) {
    console.error("[Retrieval] Error during document retrieval:", error);
    try { (getActiveLangfuseTrace() as any)?.span?.({ name: "retrieve_error" })?.end?.(); } catch {}

    if (error instanceof RetrievalError) {
      throw error;
    }

    throw new RetrievalError("Document retrieval failed", {
      query: state.query,
      originalError: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Helper function to create enhanced retrieval filters based on query analysis
 */
export const createRetrievalFilter = (query: string): Record<string, any> => {
  const queryLower = query.toLowerCase();
  const baseFilter = { program: "PGPM" };

  // Add type-based filters based on query content
  if (queryLower.includes("admission") || queryLower.includes("eligibility")) {
    return { ...baseFilter, type: "admissions" };
  }

  if (queryLower.includes("fee") || queryLower.includes("cost") || queryLower.includes("payment")) {
    return { ...baseFilter, type: "fees" };
  }

  if (queryLower.includes("placement") || queryLower.includes("job") || queryLower.includes("salary")) {
    return { ...baseFilter, type: "placements" };
  }

  if (queryLower.includes("curriculum") || queryLower.includes("course") || queryLower.includes("subject")) {
    return { ...baseFilter, type: "curriculum" };
  }

  return baseFilter;
};

/**
 * Helper function to rank and reorder documents based on relevance
 */
export const reorderDocumentsByRelevance = (docs: Document[], query: string): Document[] => {
  const queryLower = query.toLowerCase();

  return docs.sort((a, b) => {
    let scoreA = 0;
    let scoreB = 0;

    // Boost score if title/section matches query terms
    const queryTerms = queryLower.split(' ').filter(term => term.length > 2);

    for (const term of queryTerms) {
      if (a.metadata['section']?.toLowerCase().includes(term)) scoreA += 2;
      if (b.metadata['section']?.toLowerCase().includes(term)) scoreB += 2;

      if (a.pageContent.toLowerCase().includes(term)) scoreA += 1;
      if (b.pageContent.toLowerCase().includes(term)) scoreB += 1;
    }

    // Prefer certain document types based on query
    const typePreferences: Record<string, number> = {
      admissions: queryLower.includes("admission") ? 3 : 0,
      fees: queryLower.includes("fee") ? 3 : 0,
      placements: queryLower.includes("placement") ? 3 : 0,
      curriculum: queryLower.includes("curriculum") ? 3 : 0,
    };

    scoreA += typePreferences[a.metadata['type'] as string] || 0;
    scoreB += typePreferences[b.metadata['type'] as string] || 0;

    return scoreB - scoreA; // Higher scores first
  });
};

/**
 * Helper function to deduplicate similar document chunks
 */
export const deduplicateDocuments = (docs: Document[]): Document[] => {
  const seen = new Set<string>();
  const uniqueDocs: Document[] = [];

  for (const doc of docs) {
    // Create a simple hash of the content (first 200 characters)
    const contentHash = doc.pageContent.substring(0, 200).replace(/\s+/g, ' ').trim();

    if (!seen.has(contentHash)) {
      seen.add(contentHash);
      uniqueDocs.push(doc);
    }
  }

  return uniqueDocs;
};

// enhanceQuery now imported from shared/queryHeuristics

/**
 * Perform direct keyword search in vector store for specific terms
 */
const performDirectKeywordSearch = async (vectorStore: SupabaseVectorStore, term: string): Promise<Document[]> => {
  try {
    // Create a search query that embeds the exact term with context
    const contextualQuery = `SPJIMR PGPM program ${term} social projects community work`;
    
    // Perform a broader similarity search with the exact term
    const results = await vectorStore.similaritySearchWithScore(
      contextualQuery,
      10, // searches more documents
      { program: "PGPM" }
    );
    
    // Filter results to only include documents that actually contain the term
    const termRegex = new RegExp(`\\b${term}\\b`, 'i');
    const matchingDocs = results
      .filter(([doc, score]) => {
        const containsTerm = termRegex.test(doc.pageContent);
        console.log(`[Keyword Search] Doc from ${doc.metadata['source']}: contains "${term}": ${containsTerm}, score: ${score}`);
        return containsTerm;
      })
      .map(([doc, score]) => doc);
    
    console.log(`[Keyword Search] Found ${matchingDocs.length} documents containing "${term}"`);
    return matchingDocs;
    
  } catch (error) {
    console.error(`[Keyword Search] Error performing keyword search for "${term}":`, error);
    return [];
  }
};

/**
 * Adaptive confidence threshold based on query type and context quality
 */
export const getAdaptiveConfidenceThreshold = (query: string, contextContent: string): number => {
  const baseThreshold = 0.3;
  const queryLower = query.toLowerCase();
  const contextLower = contextContent.toLowerCase();
  
  // Much lower thresholds for specific information queries where we can verify answers exist
  const specificQueries = [
    { keywords: ['fee', 'cost', 'price'], indicators: ['fee', 'cost', 'rs.', 'rupees', 'lakh'], threshold: 0.2 },
    { keywords: ['duration', 'length', 'time'], indicators: ['month', 'year', 'duration'], threshold: 0.2 },
    { keywords: ['eligibility', 'requirement'], indicators: ['eligib', 'require', 'criteria'], threshold: 0.2 },
    { keywords: ['deadline', 'date'], indicators: ['deadline', 'date', 'last date'], threshold: 0.2 },
    { keywords: ['salary', 'package'], indicators: ['salary', 'lpa', 'package', 'compensation'], threshold: 0.2 },
    { keywords: ['admission', 'application'], indicators: ['admission', 'application', 'apply'], threshold: 0.2 },
    { keywords: ['curriculum', 'subject', 'course'], indicators: ['curriculum', 'subject', 'course', 'module'], threshold: 0.2 },
    { keywords: ['placement', 'job', 'career'], indicators: ['placement', 'job', 'career', 'company'], threshold: 0.2 }
  ];
  
  for (const queryType of specificQueries) {
    const hasKeyword = queryType.keywords.some(keyword => queryLower.includes(keyword));
    const hasIndicator = queryType.indicators.some(indicator => contextLower.includes(indicator));
    
    if (hasKeyword && hasIndicator) {
      console.log(`[Context Check] Using adaptive threshold ${queryType.threshold} for ${queryType.keywords[0]} query with relevant context`);
      return queryType.threshold;
    }
  }
  
  return baseThreshold;
};
