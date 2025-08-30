"use client"

import type React from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Send, Bot, User, CheckCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { useRef, useEffect, useState, memo } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

const MarkdownMemo = memo(function MarkdownMemo({ content }: { content: string }) {
  return (
    <div className="markdown-content prose prose-sm max-w-none text-foreground prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-ul:text-foreground prose-ol:text-foreground prose-li:text-foreground prose-blockquote:text-foreground prose-code:text-foreground prose-pre:text-foreground prose-p:mb-4 prose-headings:mb-3 prose-headings:mt-4 prose-ul:mb-4 prose-ol:mb-4">
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          code(rawProps) {
            const { inline, className, children, ...props } = rawProps as any
            return inline ? (
              <code className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                {children}
              </code>
            ) : (
              <pre className="bg-muted p-3 rounded-md overflow-x-auto">
                <code className="text-sm font-mono" {...props}>
                  {children}
                </code>
              </pre>
            )
          },
          blockquote({ children, ...props }) {
            return (
              <blockquote className="border-l-4 border-primary/20 pl-4 italic my-4" {...props}>
                {children}
              </blockquote>
            )
          },
          ul({ children, ...props }) {
            return <ul className="list-disc list-outside pl-6 space-y-2 my-4" {...props}>{children}</ul>
          },
          ol({ children, ...props }) {
            return <ol className="list-decimal list-outside pl-6 space-y-2 my-4" {...props}>{children}</ol>
          },
          li({ children, ...props }) {
            return <li className="leading-relaxed ml-0" {...props}>{children}</li>
          },
          p({ children, ...props }) {
            return <p className="mb-2 leading-relaxed" {...props}>{children}</p>
          },
          h1({ children, ...props }) {
            return <h1 className="text-2xl font-bold mb-4 mt-6" {...props}>{children}</h1>
          },
          h2({ children, ...props }) {
            return <h2 className="text-xl font-bold mb-3 mt-5" {...props}>{children}</h2>
          },
          h3({ children, ...props }) {
            return <h3 className="text-lg font-semibold mb-3 mt-4" {...props}>{children}</h3>
          },
          strong({ children, ...props }) {
            return <strong className="font-semibold" {...props}>{children}</strong>
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
})

const SAMPLE_QUESTIONS = [
  "What is the eligibility criteria for PGPM?",
  "How long is the PGPM program?",
  "What is the average salary after PGPM?",
  "What are the admission deadlines?",
  "Tell me about the curriculum structure",
]

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  createdAt: Date
  annotations?: { source: string }[]
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState("")
  const [sessionId, setSessionId] = useState<string | null>(null)

  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const BOTTOM_OFFSET_PX = 96

  // Auto-scroll to the bottom sentinel inside the scroll area
  useEffect(() => {
    if (!autoScroll) return
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, streamingContent, autoScroll])

  const handleScrollContainer = () => {
    const el = scrollAreaRef.current
    if (!el) return
    const proximity = el.scrollHeight - (el.scrollTop + el.clientHeight)
    setAutoScroll(proximity < 120)
  }

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Ensure input is focusable immediately after streaming and loading finish
  useEffect(() => {
    if (!isLoading && !isStreaming) {
      inputRef.current?.focus()
    }
  }, [isLoading, isStreaming])

  const handleSampleQuestion = (question: string) => {
    setInput(question)
    inputRef.current?.focus()
  }

  const focusInput = () => {
    inputRef.current?.focus()
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!input.trim() || isLoading || isStreaming) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      createdAt: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    focusInput()
    setIsLoading(true)
    setIsStreaming(true)
    setStreamingContent("")

    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      }

      // Include session ID if we have one
      if (sessionId) {
        headers["X-Session-ID"] = sessionId
      }

      // Start streaming request via Next.js proxy to avoid CORS/env issues
      const response = await fetch(`/api/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          question: userMessage.content,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to start streaming")
      }

      if (!response.body) {
        throw new Error("No response body")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullContent = ""
      let sources: string[] = []
      let buffer = ""

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })

          // Process complete SSE events separated by double newlines
          const events = buffer.split('\n\n')
          // Keep the last partial event in buffer
          buffer = events.pop() || ""

          for (const rawEvent of events) {
            // Each event may contain multiple lines; we care about data: lines
            const dataLines = rawEvent
              .split('\n')
              .filter((l) => l.startsWith('data: '))
              .map((l) => l.slice(6))
            if (dataLines.length === 0) continue

            const payload = dataLines.join('\n')
            try {
              const eventData = JSON.parse(payload)
              switch (eventData.type) {
                case 'connected':
                  if (!sessionId && eventData.sessionId) {
                    setSessionId(eventData.sessionId)
                  }
                  break
                case 'token':
                  fullContent += eventData.content
                  setStreamingContent(fullContent)
                  break
                case 'complete':
                  fullContent = eventData.answer
                  sources = eventData.sources || []
                  setStreamingContent("")
                  setIsStreaming(false)

                  // Add final assistant message
                  const assistantMessage: Message = {
                    id: (Date.now() + 1).toString(),
                    role: "assistant",
                    content: fullContent,
                    createdAt: new Date(),
                    annotations: sources.length > 0
                      ? sources.map((source: string) => ({ source }))
                      : undefined,
                  }
                  setMessages((prev) => [...prev, assistantMessage])
                  focusInput()
                  break
                case 'error':
                  console.error("Streaming error:", eventData.error)
                  setIsStreaming(false)
                  throw new Error(eventData.error)
                case 'end':
                  setIsStreaming(false)
                  focusInput()
                  break
                default:
                  // ignore other event types (e.g., pings)
                  break
              }
            } catch (parseError) {
              console.error("Error parsing SSE data:", parseError)
            }
          }
        }
      } finally {
        reader.releaseLock()
      }

    } catch (error) {
      console.error("Chat error:", error)
      setIsStreaming(false)
      setStreamingContent("")
      
      // Fallback to non-streaming
      try {
        const fallbackResponse = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(sessionId && { "X-Session-ID": sessionId }),
          },
          body: JSON.stringify({
            messages: [...messages, userMessage].map((m) => ({
              role: m.role,
              content: m.content,
            })),
          }),
        })

        if (fallbackResponse.ok) {
          const data = await fallbackResponse.json()
          
          if (!sessionId && data.sessionId) {
            setSessionId(data.sessionId)
          }

          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: data.content || "I apologize, but I couldn't process your request. Please try again.",
            createdAt: new Date(),
            annotations: data.sources && data.sources.length > 0 
              ? data.sources.map((source: string) => ({ source })) 
              : undefined,
          }
          setMessages((prev) => [...prev, assistantMessage])
        } else {
          throw new Error("Fallback request failed")
        }
      } catch (fallbackError) {
        console.error("Fallback error:", fallbackError)
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "I'm sorry, I encountered an error. Please try again later.",
          createdAt: new Date(),
        }
        setMessages((prev) => [...prev, errorMessage])
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }

  return (
    <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full px-4 py-6">
      {/* Welcome Message */}
      {messages.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center text-center space-y-6">
          <div>
            <h2 className="text-2xl font-semibold text-primary">PGPM Assistant</h2>
            <p className="text-muted-foreground max-w-md">Ask about admissions,
              curriculum & more</p>
          </div>

          <div className="space-y-3 w-full max-w-2xl">
            <p className="text-sm font-medium text-foreground">Try asking:</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {SAMPLE_QUESTIONS.map((question, index) => (
                <Button
                  key={index}
                  variant="outline"
                  className="text-left justify-start h-auto p-3 text-sm bg-transparent"
                  onClick={() => handleSampleQuestion(question)}
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      {messages.length > 0 && (
        <ScrollArea className="flex-1 pr-4 scroll-smooth" ref={scrollAreaRef} onScroll={handleScrollContainer}>
          <div className="space-y-4 pb-0">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn("flex gap-3", message.role === "user" ? "justify-end" : "justify-start")}
              >
                {message.role === "assistant" && (
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                      <Bot className="h-4 w-4 text-primary" />
                    </div>
                  </div>
                )}

                <div className={cn("max-w-[80%] space-y-2", message.role === "user" ? "items-end" : "items-start")}>
                  <Card
                    className={cn("p-4", message.role === "user" ? "bg-primary text-white" : "bg-card")}
                  >
                    <div className="space-y-2">
                      {message.role === "assistant" ? (
                        <MarkdownMemo content={message.content} />
                      ) : (
                        <p className="text-sm leading-relaxed text-white">{message.content}</p>
                      )}

                      {message.role === "assistant" && message.annotations && (
                        <div className="flex flex-wrap gap-1 pt-2 border-t border-border/50">
                          <div className="flex items-center gap-1 text-xs text-muted-foreground">
                            <CheckCircle className="h-3 w-3" />
                            Sources:
                          </div>
                          {message.annotations.map((annotation, index) => (
                            <Badge key={index} variant="secondary" className="text-xs">
                              {annotation.source || `Source ${index + 1}`}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  </Card>

                  <div className="text-xs text-muted-foreground px-1">
                    {message.createdAt.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                </div>

                {message.role === "user" && (
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-secondary/10 flex items-center justify-center">
                      <User className="h-4 w-4 text-secondary" />
                    </div>
                  </div>
                )}
              </div>
            ))}

            {(isLoading || isStreaming) && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                </div>
                <div className="max-w-[80%]">
                  <Card className="p-4 bg-card">
                  {isStreaming && streamingContent ? (
                    <>
                      <MarkdownMemo content={streamingContent} />
                      <div className="flex items-center gap-2 mt-2">
                        <div className="flex gap-1">
                          <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                          <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                          <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                        </div>
                        <span className="text-sm text-muted-foreground">Streaming...</span>
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center gap-2 animate-pulse">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                        <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                        <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                      </div>
                      <span className="text-sm text-muted-foreground">Thinking...</span>
                    </div>
                  )}
                  </Card>
                </div>
              </div>
            )}
            <div ref={bottomRef} style={{ height: BOTTOM_OFFSET_PX }} />
          </div>
        </ScrollArea>
      )}

      {/* Input Form */}
      <div className="border-t border-border pt-4 mt-4 sticky bottom-0 bg-background z-10">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            placeholder="Ask about PGPM admissions, curriculum, placements..."
            disabled={isLoading || isStreaming}
            className="flex-1"
          />
          <Button type="submit" disabled={!input.trim() || isLoading || isStreaming} className="px-4">
            <Send className="h-4 w-4" />
          </Button>
        </form>

        <p className="text-xs text-muted-foreground mt-2 text-center">
          This assistant provides information about SPJIMR PGPM program only. For official inquiries, please contact
          SPJIMR directly.
        </p>
      </div>
    </div>
  )
}
