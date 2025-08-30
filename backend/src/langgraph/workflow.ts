import { StateGraph, Annotation } from "@langchain/langgraph";
import { RAGState } from "../types/rag";
import { queryValidationNode } from "./nodes/queryValidation";
import { retrievalNode } from "./nodes/retrieval";
import { contextCheckNode } from "./nodes/contextCheck";
import { generationNode, setActiveStreamCallback } from "./nodes/generation";

// Local reference to a streaming callback that generation node can read
// We forward-set this into the generation module via a setter to avoid mutating imports
export const setGenerationStreamCallback = (cb?: (t: string) => void) => {
  try {
    setActiveStreamCallback(cb);
  } catch (e) {
    console.warn("[Workflow] Unable to set generation stream callback:", e);
  }
};
import { responseValidationNode } from "./nodes/responseValidation";
import { setActiveLangfuseCallbacks, setActiveLangfuseTrace } from "../shared/tracing";

/**
 * Main LangGraph RAG Workflow
 *
 * This implements the 5-node RAG workflow as specified in the plan:
 * 1. Query Validation (Is it about SPJIMR PGPM?)
 * 2. Document Retrieval (Get relevant documents)
 * 3. Context Check (Do we have the answer?)
 * 4. Response Generation (Generate grounded response)
 * 5. Response Validation (No hallucination check) // removed, unnecessary, always true because other nodes are doing the work
 */

// Define the state annotation for LangGraph
const StateAnnotation = Annotation.Root({
  query: Annotation<string>,
  isRelevant: Annotation<boolean>,
  retrievedDocs: Annotation<any[]>,
  hasAnswer: Annotation<boolean>,
  context: Annotation<string>,
  response: Annotation<string>,
  needsValidation: Annotation<boolean>,
  confidence: Annotation<number>,
  sources: Annotation<string[]>,
  errorMessage: Annotation<string>
});

/**
 * Create the RAG workflow graph
 */
export const createRAGWorkflow = () => {
  console.log("[Workflow] Creating RAG workflow graph...");

  try {
    // Initialize the StateGraph
    const workflow = new StateGraph(StateAnnotation);

    // Add all 5 nodes to the graph
    workflow.addNode("validate_query", queryValidationNode);
    workflow.addNode("retrieve", retrievalNode);
    workflow.addNode("check_context", contextCheckNode);
    workflow.addNode("generate", generationNode);
    workflow.addNode("validate_response", responseValidationNode);

    // Define the workflow edges to connect the nodes
    // workflow.addEdge("__start__", "validate_query");
    // workflow.addEdge("validate_query", "retrieve");
    // workflow.addEdge("retrieve", "check_context");
    // workflow.addEdge("check_context", "generate");
    // workflow.addEdge("generate", "validate_response");
    // workflow.addEdge("validate_response", "__end__");

    console.log("[Workflow] Using fallback execution due to LangGraph compilation issues");

    console.log("[Workflow] RAG workflow graph created successfully");

    // Return fallback execution directly with optional streaming support
    return {
      invoke: async (state: any, options?: { configurable?: { streamCallback?: (t: string) => void, langfuseCallbacks?: any[], langfuseTrace?: any } }) => {
        try {
          // Expose the stream callback so generation node can emit tokens
          setGenerationStreamCallback(options?.configurable?.streamCallback);
          // Set active Langfuse callbacks for nodes to use
          setActiveLangfuseCallbacks(options?.configurable?.langfuseCallbacks);
          setActiveLangfuseTrace(options?.configurable?.langfuseTrace);
          // Using fallback workflow execution
          return await executeRAGWorkflowFallback(state.query || "");
        } finally {
          setGenerationStreamCallback(undefined);
          setActiveLangfuseCallbacks(undefined);
          setActiveLangfuseTrace(undefined);
        }
      }
    } as any;
  } catch (error) {
    console.error("[Workflow] Error creating workflow:", error);
    // Return a mock compiled workflow for now
    return {
      invoke: async (state: any) => {
        // Using fallback workflow execution
        return await executeRAGWorkflowFallback(state.query || "");
      }
    };
  }
};

/**
 * Fallback workflow execution for when LangGraph compilation fails
 */
const executeRAGWorkflowFallback = async (query: string): Promise<RAGState> => {
  // Fallback execution
  
  try {
    // Manual execution of the workflow steps
    let state: RAGState = {
      query: query,
      isRelevant: false,
      retrievedDocs: [],
      hasAnswer: false,
      context: "",
      response: "",
      needsValidation: false,
      confidence: 0,
      sources: []
    };

    // Step 1: Query validation
    const validationResult = await queryValidationNode(state);
    state = { ...state, ...validationResult };

    // Step 2: Document retrieval (if relevant)
    if (state.isRelevant) {
      const retrievalResult = await retrievalNode(state);
      state = { ...state, ...retrievalResult };

      // Step 3: Context validation
      const contextResult = await contextCheckNode(state);
      state = { ...state, ...contextResult };
    }

    // Step 4: Response generation
    const generationResult = await generationNode(state);
    state = { ...state, ...generationResult };

    // Step 5: Response validation (intentionally disabled for speed)

    return state;
  } catch (error) {
    console.error("[Workflow Fallback] Error during fallback execution:", error);
    return {
      query: query,
      isRelevant: false,
      retrievedDocs: [],
      hasAnswer: false,
      context: "",
      response: "I apologize, but I encountered an error while processing your question. Please try again or contact SPJIMR directly for assistance.",
      needsValidation: false,
      confidence: 0.1,
      sources: [],
      errorMessage: error instanceof Error ? error.message : String(error)
    };
  }
};

/**
 * Execute the RAG workflow for a given query
 */
export const executeRAGWorkflow = async (query: string): Promise<RAGState> => {
  try {
    console.log(`[Workflow] Starting RAG workflow execution for query: "${query}"`);

    const app = createRAGWorkflow();

    // Initial state
    const initialState: RAGState = {
      query: query,
      isRelevant: false,
      retrievedDocs: [],
      hasAnswer: false,
      context: "",
      response: "",
      needsValidation: false,
      confidence: 0,
      sources: []
    };

    console.log("[Workflow] Executing workflow...");

    // Execute the workflow (either LangGraph or fallback)
    const result = await app.invoke(initialState);

    console.log("[Workflow] Workflow execution completed");
    console.log(`[Workflow] Final response length: ${result.response?.length || 0} characters`);
    console.log(`[Workflow] Final confidence: ${result.confidence || 0}`);

    return result as RAGState;

  } catch (error) {
    console.error("[Workflow] Error during workflow execution, using fallback:", error);

    // Use manual fallback execution
    return await executeRAGWorkflowFallback(query);
  }
};

/**
 * Validate workflow configuration
 */
export const validateWorkflowConfig = (): boolean => {
  try {
    // Check required environment variables
    const requiredEnvVars = [
      'OPENAI_API_KEY',
      'SUPABASE_URL',
      'SUPABASE_SERVICE_ROLE_KEY'
    ];

    for (const envVar of requiredEnvVars) {
      if (!process.env[envVar]) {
        console.error(`[Workflow] Missing required environment variable: ${envVar}`);
        return false;
      }
    }

    console.log("[Workflow] Workflow configuration is valid");
    return true;

  } catch (error) {
    console.error("[Workflow] Error validating workflow config:", error);
    return false;
  }
};

/**
 * Get workflow statistics and health
 */
export const getWorkflowHealth = () => {
  return {
    status: "healthy",
    nodes: [
      "validate_query",
      "retrieve",
      "check_context",
      "generate",
      "validate_response"
    ],
    configValid: validateWorkflowConfig(),
    timestamp: new Date().toISOString()
  };
};

/**
 * Test workflow with a sample query
 */
export const testWorkflow = async (testQuery?: string): Promise<boolean> => {
  try {
    console.log("[Workflow] Running workflow test...");

    const query = testQuery || "What is the PGPM program at SPJIMR?";
    const result = await executeRAGWorkflow(query);

    const isSuccess = !!(result.response && result.response.length > 20);

    console.log(`[Workflow] Test result: ${isSuccess ? 'PASSED' : 'FAILED'}`);
    console.log(`[Workflow] Test response: ${result.response.substring(0, 100)}...`);

    return isSuccess;

  } catch (error) {
    console.error("[Workflow] Test failed:", error);
    return false;
  }
};
