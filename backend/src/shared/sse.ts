import { Response } from "express";

export type SSEPayload = Record<string, any> | undefined;

export const sendEvent = (res: Response, type: string, payload?: SSEPayload) => {
  const frame = { type, ...(payload || {}) };
  res.write(`data: ${JSON.stringify(frame)}\n\n`);
  try { (res as any).flush?.(); } catch {}
};

export const sendConnected = (res: Response, sessionId: string) => {
  sendEvent(res, "connected", { sessionId, timestamp: new Date().toISOString() });
};

export const sendStatus = (res: Response, message: string) => {
  sendEvent(res, "status", { message });
};

export const sendToken = (res: Response, token: string) => {
  sendEvent(res, "token", { content: token });
};

export const sendComplete = (res: Response, data: {
  answer: string;
  confidence?: number;
  sources?: string[];
  retrievedDocuments?: number;
}) => {
  sendEvent(res, "complete", data);
};

export const sendError = (res: Response, error: string, details?: string) => {
  sendEvent(res, "error", { error, details });
};

export const sendEnd = (res: Response) => {
  sendEvent(res, "end");
};

export const startPing = (res: Response, intervalMs: number = 15000): ReturnType<typeof setInterval> => {
  return setInterval(() => {
    sendEvent(res, "ping", { t: Date.now() });
  }, intervalMs);
};

export const stopPing = (timer: ReturnType<typeof setInterval>) => {
  try { clearInterval(timer); } catch {}
};


