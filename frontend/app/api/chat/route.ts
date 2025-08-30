const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000'

export async function POST(req: Request) {
  try {
    const { messages } = await req.json()
    const lastMessage = messages[messages.length - 1]

    // Note: streaming handled via /api/chat/stream

    // Forward the request to the backend RAG API
    const response = await fetch(`${BACKEND_API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': req.headers.get('X-Session-ID') || '',
      },
      body: JSON.stringify({
        question: lastMessage.content,
        includeContext: false
      })
    })

    if (!response.ok) {
      throw new Error(`Backend API responded with status: ${response.status}`)
    }

    const data = await response.json()

    // Transform backend response to match frontend expectations
    return Response.json({
      content: data.answer || "I apologize, but I couldn't process your request. Please try again.",
      sources: data.sources || [],
      confidence: data.confidence,
      sessionId: data.sessionId
    })

  } catch (error) {
    console.error("API Proxy Error:", error)
    
    // Fallback response if backend is unavailable
    return Response.json({
      content: "I'm sorry, the RAG service is currently unavailable. Please try again later.",
      sources: [],
    }, { status: 500 })
  }
}
