# LLM Customer Support Chatbot

A production-grade customer support chatbot powered by locally-hosted LLMs via Ollama. Combines a two-stage RAG pipeline (dense retrieval + MMR reranking + LLM-augmented generation) with real-time streaming, a knowledge base management UI, feedback collection, and analytics.

---

## Architecture

```
User
 │
 ▼
React/MUI Frontend (Vite, TypeScript)
 │  Support Chat │ Admin Tab │ Analytics Panel
 │
 ▼  POST /support  (or /support-stream for SSE)
FastAPI Backend (Python 3.12, async httpx)
 │
 ├─► Query Preprocessor
 │    └─ lowercase + abbrev expansion + punctuation strip
 │    └─ LLM query expansion → 3 alternative phrasings
 │    └─ Embed all 4 variants → averaged embedding
 │
 ├─► Stage 1: Dense Retrieval (nomic-embed-text via Ollama)
 │    └─ Cosine similarity search → top-K candidates (default K=5)
 │
 ├─► Stage 2: MMR Reranking
 │    └─ Maximal Marginal Relevance → top-3 diverse results
 │
 ├─► Confidence check (threshold, default 0.75)
 │    ├─ Hit  → LLM-augmented answer (Mistral + top-3 context)
 │    └─ Miss → Direct LLM generation (Mistral fallback)
 │
 └─► Response + session_id + confidence score
      │
      ├─► ConversationMemory (per session, sliding window of 10)
      ├─► AnalyticsStore (query_log.json)
      └─► FeedbackStore (feedback.json) ← POST /feedback
```

---

## Quickstart — Local Development

**Prerequisites:** Python 3.12+, Node 20+, [Ollama](https://ollama.com) installed and running.

```bash
# 1. Pull required models
ollama pull mistral
ollama pull nomic-embed-text

# 2. Backend
cd server/API
cp .env.example .env          # edit if needed
pip install -r ../Docker_api/requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# 3. Frontend (new terminal)
cd client/llm-client
cp .env.example .env          # edit VITE_API_URL if needed
npm install
npm run dev
# → http://localhost:5173
```

---

## Quickstart — Docker

```bash
cd server/Docker_api
cp ../../client/llm-client/.env.example .env   # optional overrides
docker compose up --build
```

Expected output:
```
ollama     | Ollama is running
api        | Pulling model: mistral...
api        | Pulling model: nomic-embed-text...
api        | INFO: Application startup complete.
client     | nginx: ready
```

- Frontend: http://localhost:80
- API:      http://localhost:8002
- Ollama:   http://localhost:11434

The `api` service waits for Ollama's healthcheck before starting. Models are pulled automatically on first run and cached in the `ollama_data` Docker volume.

---

## Environment Variables

### Server (`server/API/.env.example`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated CORS origins |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity for a RAG hit |
| `TOP_K` | `5` | Number of candidates retrieved before MMR reranking |
| `MIN_CONFIDENCE` | `media` | Minimum confidence label (`alta`/`media`/`baja`) |
| `ADMIN_PASSWORD` | `admin123` | Password for `/knowledge-base` endpoints |
| `VECTOR_STORE_PATH` | `./vector_store.json` | Path to persisted vector store |
| `FEEDBACK_FILE` | `./feedback.json` | Path to feedback log |
| `QUERY_LOG_FILE` | `./query_log.json` | Path to analytics query log |

### Client (`client/llm-client/.env.example`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8002` | Backend API URL |
| `VITE_ADMIN_PASSWORD` | _(empty)_ | Pre-fills admin password field (optional) |

---

## API Reference

All endpoints are on the FastAPI server (default port 8002). Interactive docs at `/docs`.

### Core

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | — | API info |
| `GET` | `/health` | — | Status, Ollama reachability, vector store size, uptime |
| `GET` | `/system-info` | — | CPU/memory/GPU usage |
| `POST` | `/support` | — | Main support endpoint (RAG pipeline) |
| `POST` | `/support-stream` | — | Streaming support via SSE |
| `POST` | `/generate` | — | Direct LLM generation with context |
| `POST` | `/embeddings` | — | Embed arbitrary texts |
| `POST` | `/train-support` | — | Load Q&A cases into the knowledge base |
| `POST` | `/get-similar-cases` | — | Raw vector search (debug/inspection) |

### Knowledge Base Management

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/knowledge-base` | `X-Admin-Password` | List all cases (id, category, question, created_at) |
| `DELETE` | `/knowledge-base/{case_id}` | `X-Admin-Password` | Remove a case by ID |

### Feedback & Analytics

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/feedback` | — | Submit rating (1-5) for a message |
| `GET` | `/feedback/stats` | — | Aggregated feedback statistics |
| `GET` | `/analytics` | — | Query stats, RAG hit rate, top categories, Ollama status |

---

### Example: Train the system

```bash
curl -X POST http://localhost:8002/train-support \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [
      {
        "question": "Where is my order?",
        "answer": "Please share your order ID and I will track it.",
        "category": "seguimiento_pedido",
        "priority": 1
      }
    ],
    "use_gpu": false
  }'
```

### Example: Ask a question (non-streaming)

```bash
curl -X POST http://localhost:8002/support \
  -H "Content-Type: application/json" \
  -d '{"text": "My package has not arrived", "use_gpu": false}'
```

Response:
```json
{
  "response": "I'm sorry to hear that! Could you share your order ID so I can track it?",
  "session_id": "3f8a1c2d-...",
  "confidence": 0.87,
  "rag_hit": true,
  "cases_used": 3,
  "response_time_ms": 842
}
```

### Example: Streaming (SSE)

```bash
curl -X POST http://localhost:8002/support-stream \
  -H "Content-Type: application/json" \
  -d '{"text": "I need a refund", "use_gpu": false}'
```

SSE events:
```
data: {"type":"metadata","session_id":"...","confidence":0.81,"rag_hit":true}

data: "Sure"
data: ","
data: " I"
data: " can"
...
data: [DONE]
```

### Example: Submit feedback

```bash
curl -X POST http://localhost:8002/feedback \
  -H "Content-Type: application/json" \
  -d '{"message_id": "3f8a1c2d", "rating": 5, "comment": "Very helpful!"}'
```

### Example: Admin — list knowledge base

```bash
curl http://localhost:8002/knowledge-base \
  -H "X-Admin-Password: admin123"
```

### Example: Admin — delete a case

```bash
curl -X DELETE http://localhost:8002/knowledge-base/<case_id> \
  -H "X-Admin-Password: admin123"
```

---

## RAG Pipeline Explained

### 1. Query Preprocessing
Before embedding, the user query is:
- Lowercased and punctuation-stripped (preserving emails and `ORDxxxxxx` patterns)
- Expanded for common abbreviations (`don't → do not`, etc.)

### 2. Query Expansion
The preprocessed query is sent to Mistral with the prompt:
> *"Generate 3 alternative phrasings of this customer support query..."*

All 4 texts (original + 3 alternatives) are embedded using `nomic-embed-text` and averaged into a single representative embedding. This dramatically improves recall for short or ambiguous queries.

### 3. Dense Retrieval
The averaged embedding is compared against all stored case embeddings using cosine similarity. The top-K candidates (default K=5) are returned.

### 4. MMR Reranking
Maximal Marginal Relevance selects the top-3 candidates that maximize:
```
score(d) = λ · sim(d, query) − (1 − λ) · max(sim(d, selected))
```
with λ=0.7. This balances relevance with diversity, avoiding redundant answers.

### 5. LLM-Augmented Generation
- **RAG hit** (confidence ≥ threshold): The top-3 cases are formatted as a context block and passed to Mistral with a support agent system prompt. The LLM generates a natural, conversational response grounded in the knowledge base.
- **RAG miss**: Mistral generates a free-form response using the conversation history only.

### Conversation Memory
Each session maintains a sliding window of 10 messages (user + assistant), tracking which RAG cases were used per turn. This context is included in every LLM call.

---

## Project Structure

```
LLM/
├── README.md
├── client/
│   └── llm-client/
│       ├── Dockerfile              # nginx multi-stage build
│       ├── .env.example
│       ├── vite.config.ts          # exposes VITE_API_URL, VITE_ADMIN_PASSWORD
│       └── src/
│           ├── App.tsx             # MUI ThemeProvider
│           └── screens/
│               └── ChatInterface.tsx  # Chat, Admin tab, Analytics panel, SSE streaming
│
└── server/
    ├── API/                        # Full-featured application (source of truth)
    │   ├── main.py                 # FastAPI app — all 15 endpoints
    │   ├── vector_store.py         # Async persistent vector store + MMR reranking
    │   ├── support_trainer.py      # Two-stage retrieval wrapper
    │   ├── query_processor.py      # Preprocessing + query expansion + multi-embedding
    │   ├── conversation_memory.py  # Per-session sliding window memory
    │   ├── feedback_store.py       # Feedback persistence (feedback.json)
    │   ├── analytics_store.py      # Query log + stats (query_log.json)
    │   ├── support_models.py       # Pydantic models
    │   ├── simulated_orders.py     # In-memory order database (100 fake orders)
    │   ├── conftest.py             # Pytest fixtures
    │   ├── test_support.py         # Pytest test suite
    │   └── .env.example
    │
    └── Docker_api/                 # Docker configuration
        ├── Dockerfile              # python:3.12-slim, copies from server/API/
        ├── docker-compose.yml      # ollama + api + client, proper healthchecks
        ├── init.sh                 # Waits for Ollama, pulls models, starts uvicorn
        └── requirements.txt        # Pinned Python deps
```

---

## Running Tests

```bash
cd server/API
pip install pytest httpx
pytest test_support.py -v

# Tests that require Ollama are skipped automatically if it is not running.
# To run all tests including Ollama-dependent ones:
ollama serve &
pytest test_support.py -v
```
