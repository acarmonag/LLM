import os
import re
import json
import time
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import psutil
import GPUtil
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from support_models import SupportCase, SupportEmbeddingInput, SupportConfig
from support_trainer import SupportTrainer
from vector_store import VectorStore
from query_processor import QueryProcessor
from conversation_memory import ConversationStore
from simulated_orders import OrderDatabase
from feedback_store import FeedbackStore
from analytics_store import AnalyticsStore, QueryLog

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_CONFIDENCE = os.getenv("MIN_CONFIDENCE", "media")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
VECTOR_STORE_PATH = os.getenv(
    "VECTOR_STORE_PATH", os.path.join(os.path.dirname(__file__), "vector_store.json")
)

# ---------------------------------------------------------------------------
# Services (module-level singletons)
# ---------------------------------------------------------------------------
config = SupportConfig(threshold=SIMILARITY_THRESHOLD, top_k=TOP_K, min_confidence=MIN_CONFIDENCE)
vector_store = VectorStore(persist_path=VECTOR_STORE_PATH)
trainer = SupportTrainer(config=config, vector_store=vector_store)
order_db = OrderDatabase()
query_processor = QueryProcessor()
conversation_store = ConversationStore()
feedback_store = FeedbackStore()
analytics_store = AnalyticsStore()

http_client: httpx.AsyncClient = None
_startup_time: float = time.time()

SUPPORT_SYSTEM_PROMPT = (
    "You are a helpful customer support agent. Use ONLY the provided knowledge base "
    "context to answer accurately and conversationally. If the context does not contain "
    "the answer, say so clearly. Keep your response concise and friendly."
)

logger.info(json.dumps({"msg": f"LLM model: {LLM_MODEL}", "ollama_url": OLLAMA_URL}))

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, _startup_time
    _startup_time = time.time()
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    await vector_store.load()
    yield
    await vector_store.save()
    if http_client:
        await http_client.aclose()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Customer Support API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        json.dumps({
            "request_id": request_id,
            "method": request.method,
            "endpoint": str(request.url.path),
            "status_code": response.status_code,
            "response_time_ms": elapsed_ms,
        })
    )
    response.headers["X-Request-ID"] = request_id
    return response

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class QueryInput(BaseModel):
    text: str
    max_length: int = 50
    use_gpu: Optional[bool] = False


class EmbeddingInput(BaseModel):
    texts: List[str]
    use_gpu: Optional[bool] = False


class MessageContext(BaseModel):
    role: str
    content: str


class GenerateWithContext(BaseModel):
    text: str
    context: List[MessageContext] = []
    max_length: int = 50
    use_gpu: Optional[bool] = False


class SupportQuery(BaseModel):
    text: str
    session_id: Optional[str] = None
    max_length: int = 50
    use_gpu: Optional[bool] = False


class FeedbackInput(BaseModel):
    message_id: str
    rating: int  # 1-5
    comment: Optional[str] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_system_info() -> dict:
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                "id": gpu.id,
                "name": gpu.name,
                "load": f"{gpu.load * 100:.1f}%",
                "memory_used": f"{gpu.memoryUsed}MB",
                "memory_total": f"{gpu.memoryTotal}MB",
                "temperature": f"{gpu.temperature}C",
            })
    except Exception as e:
        logger.warning(json.dumps({"msg": f"GPU info unavailable: {e}"}))
    return {
        "cpu_usage": f"{cpu_percent}%",
        "memory_used": f"{memory.percent}%",
        "gpu_info": gpu_info,
    }


async def call_ollama_generate(prompt: str, system: str = None) -> str:
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    response = await http_client.post(
        f"{OLLAMA_URL}/api/generate", json=payload, timeout=60.0
    )
    if response.status_code != 200:
        raise Exception(f"Ollama generate failed: {response.status_code}")
    return response.json().get("response", "").strip()


async def call_ollama_generate_stream(prompt: str, system: str = None):
    """Yield tokens from Ollama streaming generate."""
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": True}
    if system:
        payload["system"] = system
    async with http_client.stream(
        "POST", f"{OLLAMA_URL}/api/generate", json=payload, timeout=60.0
    ) as response:
        if response.status_code != 200:
            raise Exception(f"Ollama stream failed: {response.status_code}")
        async for line in response.aiter_lines():
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue


async def get_embedding(text: str) -> list[float]:
    payload = {"model": EMBEDDING_MODEL, "prompt": text}
    response = await http_client.post(
        f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=30.0
    )
    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.status_code}")
    return response.json()["embedding"]


def build_rag_context(cases: list[dict]) -> str:
    parts = []
    for i, case in enumerate(cases, 1):
        c = case.get("case", case)
        parts.append(f"Case {i}: Q: {c.get('question', '')} A: {c.get('answer', '')}")
    return "\n".join(parts)


def require_admin(x_admin_password: Optional[str]) -> None:
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")


async def _check_ollama() -> bool:
    try:
        r = await http_client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "LLM Customer Support API v2.0", "llm_model": LLM_MODEL}


@app.get("/health")
async def health():
    ollama_ok = await _check_ollama()
    return {
        "status": "ok",
        "ollama_reachable": ollama_ok,
        "vector_store_size": vector_store.size,
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "version": "2.0.0",
    }


@app.get("/system-info")
async def system_info():
    return get_system_info()


@app.post("/generate")
async def generate_text(query: GenerateWithContext):
    try:
        context_string = "\n".join(
            f"{msg.role}: {msg.content}" for msg in query.context[-5:]
        )
        full_prompt = (
            f"{context_string}\nUsuario: {query.text}\nAsistente:"
            if context_string
            else f"Usuario: {query.text}\nAsistente:"
        )
        generated_text = await call_ollama_generate(full_prompt)
        return {"generated_text": generated_text, "system_info": get_system_info()}
    except Exception as e:
        logger.error(json.dumps({"msg": f"generate error: {e}"}))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings")
async def get_embeddings(input_data: EmbeddingInput):
    try:
        results = []
        for text in input_data.texts:
            payload = {"model": EMBEDDING_MODEL, "prompt": text}
            response = await http_client.post(
                f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=30.0
            )
            if response.status_code != 200:
                raise Exception(f"Embedding failed: {response.status_code}: {response.text}")
            results.append(response.json())
        return {"embeddings": results, "system_info": get_system_info()}
    except Exception as e:
        logger.error(json.dumps({"msg": f"embeddings error: {e}"}))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/support-embeddings")
async def get_support_embeddings(input_data: SupportEmbeddingInput):
    try:
        results = []
        for case in input_data.cases:
            combined = f"Q: {case.question}\nA: {case.answer}\nCategory: {case.category}"
            payload = {"model": EMBEDDING_MODEL, "prompt": combined}
            response = await http_client.post(
                f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=30.0
            )
            if response.status_code != 200:
                raise Exception(f"Embedding failed: {response.status_code}")
            results.append({"case": case.dict(), "embedding": response.json()["embedding"]})
        return {
            "message": f"Generated embeddings for {len(results)} support cases",
            "embeddings": results,
        }
    except Exception as e:
        logger.error(json.dumps({"msg": f"support-embeddings error: {e}"}))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-support")
async def train_support_system(input_data: SupportEmbeddingInput):
    embeddings_response = await get_support_embeddings(input_data)
    await trainer.add_cases_async(embeddings_response["embeddings"])
    await vector_store.save()
    return {"message": "Training data added successfully", "cases_count": vector_store.size}


async def _run_rag_pipeline(query_text: str, session_id: Optional[str] = None):
    """Core RAG pipeline shared by /support and /support-stream."""
    session_id, memory = conversation_store.get_or_create(session_id)
    memory.add("user", query_text)

    processed_query = await query_processor.preprocess(query_text)

    # Order lookup
    order_id_match = re.search(r'ORD\d{6}', query_text)
    order_info = None
    if order_id_match:
        order_id = order_id_match.group()
        order_info = order_db.get_order(order_id)
        if order_info:
            processed_query = (
                f"{processed_query} Orden: {order_id} Estado: {order_info['status']}"
            )

    # Query expansion + multi-embedding
    expanded = await query_processor.expand_query(processed_query, OLLAMA_URL, http_client)
    query_embedding = await query_processor.get_multi_embedding(expanded, OLLAMA_URL, http_client)

    # Two-stage retrieval
    similar_cases = await trainer.find_similar_cases_async(query_embedding)

    top_confidence = similar_cases[0]["similarity"] if similar_cases else 0.0
    rag_hit = top_confidence >= SIMILARITY_THRESHOLD
    rag_case_ids = [c.get("case_id", "") for c in similar_cases]

    # Enrich order data if applicable
    if order_info and similar_cases:
        for case in similar_cases:
            category = case["case"]["category"]
            if category in ("seguimiento_pedido", "seguimiento_detallado"):
                status_details = order_db._get_status_details(order_info)
                case["case"]["answer"] = case["case"]["answer"].replace(
                    "[Detalles especificos seran insertados dinamicamente]",
                    f"Estado actual: {order_info['status']}, "
                    f"Fecha de orden: {status_details['order_date']}, "
                    f"Total: ${status_details['total']:.2f}",
                )

    # Build LLM prompt
    conversation_context = "\n".join(
        f"{m['role']}: {m['content']}" for m in memory.to_llm_messages()[-5:]
    )

    if rag_hit and similar_cases:
        context_block = build_rag_context(similar_cases)
        prompt = (
            f"Knowledge base context:\n{context_block}\n\n"
            f"Conversation history:\n{conversation_context}\n\n"
            f"Customer query: {query_text}\n\n"
            f"Please provide a helpful response based on the knowledge base context above."
        )
    else:
        prompt = (
            f"Conversation history:\n{conversation_context}\n\n"
            f"Customer query: {query_text}\n\n"
            f"Please provide a helpful customer support response."
        )

    return session_id, prompt, top_confidence, rag_hit, rag_case_ids, similar_cases, memory


@app.post("/support")
async def support_endpoint(query: SupportQuery):
    start_time = time.time()
    try:
        session_id, prompt, top_confidence, rag_hit, rag_case_ids, similar_cases, memory = (
            await _run_rag_pipeline(query.text, query.session_id)
        )

        response_text = await call_ollama_generate(prompt, system=SUPPORT_SYSTEM_PROMPT)
        memory.add("assistant", response_text, rag_cases_used=rag_case_ids)

        response_time_ms = int((time.time() - start_time) * 1000)

        matched_category = similar_cases[0]["case"]["category"] if rag_hit and similar_cases else None
        analytics_store.log_query(QueryLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query.text,
            matched_category=matched_category,
            confidence=top_confidence,
            response_time_ms=response_time_ms,
            session_id=session_id,
            rag_hit=rag_hit,
        ))

        return {
            "response": response_text,
            "session_id": session_id,
            "confidence": top_confidence,
            "rag_hit": rag_hit,
            "cases_used": len(similar_cases),
            "response_time_ms": response_time_ms,
        }
    except Exception as e:
        logger.error(json.dumps({"msg": f"support error: {e}"}))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/support-stream")
async def support_stream_endpoint(query: SupportQuery):
    """SSE streaming response. First event is metadata, then tokens, then [DONE]."""
    start_time = time.time()

    try:
        session_id, prompt, top_confidence, rag_hit, rag_case_ids, similar_cases, memory = (
            await _run_rag_pipeline(query.text, query.session_id)
        )
    except Exception as e:
        logger.error(json.dumps({"msg": f"support-stream pipeline error: {e}"}))
        raise HTTPException(status_code=500, detail=str(e))

    async def event_generator():
        metadata = json.dumps({
            "type": "metadata",
            "session_id": session_id,
            "confidence": top_confidence,
            "rag_hit": rag_hit,
        })
        yield f"data: {metadata}\n\n"

        full_response = []
        try:
            async for token in call_ollama_generate_stream(prompt, system=SUPPORT_SYSTEM_PROMPT):
                full_response.append(token)
                yield f"data: {json.dumps(token)}\n\n"
        except Exception as e:
            logger.error(json.dumps({"msg": f"streaming error: {e}"}))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

        response_text = "".join(full_response)
        memory.add("assistant", response_text, rag_cases_used=rag_case_ids)
        response_time_ms = int((time.time() - start_time) * 1000)
        matched_category = similar_cases[0]["case"]["category"] if rag_hit and similar_cases else None
        analytics_store.log_query(QueryLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query.text,
            matched_category=matched_category,
            confidence=top_confidence,
            response_time_ms=response_time_ms,
            session_id=session_id,
            rag_hit=rag_hit,
        ))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/get-similar-cases")
async def find_similar_cases(query: QueryInput):
    """Legacy endpoint for direct vector search inspection."""
    try:
        processed_query = trainer.preprocess_text(query.text)

        order_id_match = re.search(r'ORD\d{6}', query.text)
        order_info = None
        if order_id_match:
            order_id = order_id_match.group()
            order_info = order_db.get_order(order_id)
            if order_info:
                processed_query = f"{processed_query} Orden: {order_id} Estado: {order_info['status']}"

        query_embedding = await get_embedding(processed_query)
        similar_cases = await trainer.find_similar_cases_async(query_embedding)

        if order_info and similar_cases:
            for case in similar_cases:
                if case["case"]["category"] in ("seguimiento_pedido", "seguimiento_detallado"):
                    status_details = order_db._get_status_details(order_info)
                    case["case"]["answer"] = case["case"]["answer"].replace(
                        "[Detalles especificos seran insertados dinamicamente]",
                        f"Estado actual: {order_info['status']}, "
                        f"Fecha de orden: {status_details['order_date']}, "
                        f"Total: ${status_details['total']:.2f}",
                    )

        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', query.text)
        if email_match and similar_cases:
            customer_orders = order_db.get_customer_orders(email_match.group())
            if customer_orders:
                orders_summary = "\n".join(
                    f"- Orden {o['order_id']}: {o['status']}" for o in customer_orders
                )
                similar_cases[0]["case"]["answer"] += f"\n\nOrdenes encontradas:\n{orders_summary}"

        return {
            "similar_cases": similar_cases,
            "query_processed": processed_query,
            "threshold_used": trainer.threshold,
            "total_cases": vector_store.size,
            "order_info": order_info,
            "confidence_level": similar_cases[0]["confidence"] if similar_cases else "baja",
        }
    except Exception as e:
        logger.error(json.dumps({"msg": f"get-similar-cases error: {e}"}))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Knowledge base management
# ---------------------------------------------------------------------------

@app.get("/knowledge-base")
async def get_knowledge_base(x_admin_password: Optional[str] = Header(None)):
    require_admin(x_admin_password)
    cases = await vector_store.get_all_cases()
    return {
        "cases": [
            {
                "case_id": c["case_id"],
                "category": c["category"],
                "question": c["question"],
                "created_at": c["created_at"],
            }
            for c in cases
        ],
        "total": len(cases),
    }


@app.delete("/knowledge-base/{case_id}")
async def delete_knowledge_base_case(
    case_id: str, x_admin_password: Optional[str] = Header(None)
):
    require_admin(x_admin_password)
    deleted = await vector_store.delete_case(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    await vector_store.save()
    return {"success": True, "deleted_case_id": case_id}


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.post("/feedback")
async def submit_feedback(body: FeedbackInput):
    if not (1 <= body.rating <= 5):
        raise HTTPException(status_code=422, detail="Rating must be between 1 and 5")
    entry = feedback_store.add_feedback(body.message_id, body.rating, body.comment)
    return {"success": True, "feedback": entry}


@app.get("/feedback/stats")
async def get_feedback_stats():
    return feedback_store.get_stats()


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.get("/analytics")
async def get_analytics():
    ollama_ok = await _check_ollama()
    return analytics_store.get_stats(ollama_reachable=ollama_ok)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
