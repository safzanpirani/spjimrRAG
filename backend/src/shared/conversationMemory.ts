export class ConversationMemory {
  private history: Array<{
    timestamp: Date;
    question: string;
    answer: string;
    confidence: number;
    sources: string[];
  }> = [];

  private readonly maxHistoryLength: number = 10;

  addTurn(turn: {
    question: string;
    answer: string;
    confidence: number;
    sources: string[];
  }): void {
    this.history.push({ ...turn, timestamp: new Date() });
    if (this.history.length > this.maxHistoryLength) {
      this.history.shift();
    }
  }

  getHistory() {
    return [...this.history];
  }

  getRecentContext(turnCount: number = 3): string {
    const recentTurns = this.history.slice(-turnCount);
    return recentTurns
      .map(turn => `Q: ${turn.question}\nA: ${turn.answer}`)
      .join('\n\n');
  }

  clear(): void {
    this.history = [];
  }

  getStats(): {
    totalTurns: number;
    averageConfidence: number;
    uniqueSources: number;
    timeSpan: number;
  } {
    if (this.history.length === 0) {
      return { totalTurns: 0, averageConfidence: 0, uniqueSources: 0, timeSpan: 0 };
    }

    const averageConfidence = this.history.reduce((sum, turn) => sum + turn.confidence, 0) / this.history.length;
    const uniqueSources = new Set(this.history.flatMap(turn => turn.sources)).size;
    const timeSpan = this.history.length > 1
      ? (this.history[this.history.length - 1]!.timestamp.getTime() - this.history[0]!.timestamp.getTime()) / (1000 * 60)
      : 0;

    return {
      totalTurns: this.history.length,
      averageConfidence: Math.round(averageConfidence * 1000) / 1000,
      uniqueSources,
      timeSpan: Math.round(timeSpan * 100) / 100
    };
  }
}


