"""
Pytest test suite for the LLM Customer Support API.

Run:
    cd server/API
    pytest test_support.py -v

Note: tests that call Ollama (embeddings, /support, /support-stream, training)
are marked with @pytest.mark.ollama and will be skipped automatically if
Ollama is not reachable (controlled by the trained_client fixture in conftest.py).
"""
import os
import pytest


# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------

def test_root_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health_fields(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "ollama_reachable" in body
    assert "vector_store_size" in body
    assert "uptime_seconds" in body
    assert "version" in body


def test_health_uptime_increases(client):
    import time
    r1 = client.get("/health")
    time.sleep(0.1)
    r2 = client.get("/health")
    assert r2.json()["uptime_seconds"] >= r1.json()["uptime_seconds"]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def test_submit_feedback(client):
    r = client.post("/feedback", json={"message_id": "msg-001", "rating": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["feedback"]["rating"] == 4


def test_submit_feedback_with_comment(client):
    r = client.post("/feedback", json={
        "message_id": "msg-002", "rating": 2, "comment": "Too slow"
    })
    assert r.status_code == 200
    assert r.json()["feedback"]["comment"] == "Too slow"


def test_feedback_rating_validation(client):
    r = client.post("/feedback", json={"message_id": "msg-bad", "rating": 6})
    assert r.status_code == 422


def test_feedback_stats(client):
    r = client.get("/feedback/stats")
    assert r.status_code == 200
    body = r.json()
    assert "total_count" in body
    assert "average_rating" in body
    assert "low_rated_messages" in body


def test_feedback_stats_has_low_rated(client):
    # We submitted rating=2 in test_submit_feedback_with_comment above
    r = client.get("/feedback/stats")
    stats = r.json()
    if stats["total_count"] > 0:
        assert isinstance(stats["low_rated_messages"], list)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def test_analytics_fields(client):
    r = client.get("/analytics")
    assert r.status_code == 200
    body = r.json()
    assert "total_queries_today" in body
    assert "rag_hit_rate_pct" in body
    assert "avg_confidence" in body
    assert "top_categories" in body
    assert "ollama_reachable" in body


# ---------------------------------------------------------------------------
# Knowledge base — auth
# ---------------------------------------------------------------------------

def test_knowledge_base_requires_auth(client):
    r = client.get("/knowledge-base")
    assert r.status_code == 401


def test_knowledge_base_wrong_password(client):
    r = client.get("/knowledge-base", headers={"X-Admin-Password": "wrong"})
    assert r.status_code == 401


def test_knowledge_base_correct_password_empty(client):
    r = client.get("/knowledge-base", headers={"X-Admin-Password": "testpass"})
    assert r.status_code == 200
    body = r.json()
    assert "cases" in body
    assert "total" in body
    assert isinstance(body["cases"], list)


def test_delete_nonexistent_case(client):
    r = client.delete(
        "/knowledge-base/00000000-0000-0000-0000-000000000000",
        headers={"X-Admin-Password": "testpass"},
    )
    assert r.status_code == 404


def test_delete_requires_auth(client):
    r = client.delete("/knowledge-base/some-id")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Ollama-dependent tests (skipped if Ollama unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("trained_client")
class TestWithTraining:

    def test_train_support_returns_count(self, trained_client):
        cases = {
            "cases": [{
                "question": "How do I cancel my order?",
                "answer": "You can cancel before it ships. Share your order ID.",
                "category": "cancelacion_pedido",
                "priority": 1,
            }],
            "use_gpu": False,
        }
        r = trained_client.post("/train-support", json=cases)
        assert r.status_code == 200
        assert r.json()["cases_count"] >= 1

    def test_knowledge_base_returns_cases_after_training(self, trained_client):
        r = trained_client.get(
            "/knowledge-base", headers={"X-Admin-Password": "testpass"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 3
        case = body["cases"][0]
        assert "case_id" in case
        assert "category" in case
        assert "question" in case
        assert "created_at" in case

    def test_delete_existing_case(self, trained_client):
        # Get a case to delete
        r = trained_client.get(
            "/knowledge-base", headers={"X-Admin-Password": "testpass"}
        )
        cases = r.json()["cases"]
        if not cases:
            pytest.skip("No cases to delete")
        case_id = cases[-1]["case_id"]
        r = trained_client.delete(
            f"/knowledge-base/{case_id}",
            headers={"X-Admin-Password": "testpass"},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_support_no_session(self, trained_client):
        r = trained_client.post("/support", json={"text": "Where is my order?", "use_gpu": False})
        assert r.status_code == 200
        body = r.json()
        assert "response" in body
        assert body["response"] != ""
        assert "session_id" in body
        assert "confidence" in body
        assert "rag_hit" in body

    def test_support_session_continuity(self, trained_client):
        r1 = trained_client.post("/support", json={"text": "Hello", "use_gpu": False})
        session_id = r1.json()["session_id"]
        r2 = trained_client.post("/support", json={"text": "My order is late", "session_id": session_id, "use_gpu": False})
        assert r2.json()["session_id"] == session_id

    def test_support_stream_returns_done(self, trained_client):
        import json as _json
        r = trained_client.post(
            "/support-stream",
            json={"text": "I need a refund", "use_gpu": False},
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        content = r.text
        assert "[DONE]" in content
        # First event must be metadata
        first_line = [l for l in content.split("\n") if l.startswith("data: ")][0]
        first_payload = _json.loads(first_line[6:])
        assert first_payload.get("type") == "metadata"
        assert "session_id" in first_payload

    def test_embeddings_returns_floats(self, trained_client):
        r = trained_client.post("/embeddings", json={"texts": ["hello world"], "use_gpu": False})
        assert r.status_code == 200
        embedding = r.json()["embeddings"][0]["embedding"]
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding[:5])
