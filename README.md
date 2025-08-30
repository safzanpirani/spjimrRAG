SPJIMR PGPM RAG Assistant
=========================

A full‑stack application that answers questions about SPJIMR PGPM using a LangGraph workflow on the backend and a Next.js UI on the frontend.

Quick start
-----------

1) Backend (Express + LangGraph)

- Prereqs: Node 18+, npm; Supabase project with pgvector; OpenAI API key.
- Copy `.env.example` if present or set these env vars:
  - `OPENAI_API_KEY`
  - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (and optionally `SUPABASE_ANON_KEY`)
  - `EMBEDDING_MODEL=text-embedding-3-small`
  - `FRONTEND_URL=http://localhost:3000`
  - Optional Langfuse: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`

Commands (from `backend/`):

```bash
npm i
npm run build
npm run start   # starts on http://localhost:8000
```

Useful endpoints:
- `POST /api/chat` (non‑streaming)
- `POST /api/chat/stream` (SSE streaming)
- `GET /api/health` and `/api/status`

2) Frontend (Next.js)

- Configure the backend URL for the built‑in API proxy:
  - `BACKEND_API_URL=http://localhost:8000`

Commands (from `frontend/`):

```bash
npm i
npm run dev    # http://localhost:3000
```

Architecture highlights
-----------------------

- Single LangGraph pipeline: `validate_query → retrieve → check_context → generate`.
- Streaming via SSE; conversation memory maintained per session.
- Supabase + pgvector for retrieval; GPT‑4o‑mini for generation.
- Langfuse tracing enabled: each node appears as its own step; root trace named `spjimr-rag`.

Notes
-----

- In dev, run backend first, then frontend. The frontend proxies `/api/chat/stream` to the backend using `BACKEND_API_URL`.
- If you enable Langfuse, open the trace to inspect node‑level inputs/outputs and final response.
