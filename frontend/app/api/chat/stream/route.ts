const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000'

export async function POST(req: Request) {
  try {
    const sessionId = req.headers.get('X-Session-ID') || ''
    const bodyText = await req.text()

    const backendResponse = await fetch(`${BACKEND_API_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': sessionId,
      },
      body: bodyText,
    })

    if (!backendResponse.ok || !backendResponse.body) {
      return new Response(
        JSON.stringify({ error: 'Upstream streaming failed' }),
        { status: backendResponse.status || 502 }
      )
    }

    const stream = new ReadableStream({
      start(controller) {
        const reader = backendResponse.body!.getReader()

        const pump = () => reader.read().then(({ done, value }) => {
          if (done) {
            controller.close()
            return
          }
          controller.enqueue(value)
          pump()
        }).catch((err) => {
          controller.error(err)
        })

        pump()
      }
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      }
    })

  } catch (error) {
    console.error('Stream proxy error:', error)
    return new Response(JSON.stringify({ error: 'Stream proxy error' }), { status: 500 })
  }
}


