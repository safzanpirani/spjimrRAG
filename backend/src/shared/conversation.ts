import { ConversationMemory } from "./conversationMemory";
import { isFollowUpQuery, isKnownPGPMTerm } from "./queryHeuristics";

export const buildCompositeQuery = (question: string, memory: ConversationMemory): string => {
  const q = (question || '').trim();
  if (!q) return q;

  // Specific/known terms: keep as-is
  if (isKnownPGPMTerm(q)) {
    return q;
  }

  if (isFollowUpQuery(q)) {
    const recentContext = memory.getRecentContext(5);
    return recentContext
      ? `Previous conversation context:\n${recentContext}\n\nCurrent user request: ${q}\n\nNote: This appears to be a follow-up query asking for more detailed information about the previously discussed SPJIMR PGPM topics.`
      : `User request: ${q} (Note: This appears to be a request for detailed SPJIMR PGPM information)`;
  }

  const recentContext = memory.getRecentContext(2);
  return recentContext ? `${recentContext}\n\nUser: ${q}` : q;
};


