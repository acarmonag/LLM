"""Pytest fixtures for the LLM support API test suite."""
import os
import pytest
from fastapi.testclient import TestClient

# Point stores to temp paths so tests don't pollute real data
os.environ.setdefault("VECTOR_STORE_PATH", "/tmp/test_vector_store.json")
os.environ.setdefault("FEEDBACK_FILE", "/tmp/test_feedback.json")
os.environ.setdefault("QUERY_LOG_FILE", "/tmp/test_query_log.json")
os.environ.setdefault("ADMIN_PASSWORD", "testpass")

# Import app AFTER env vars are set
from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Bare test client — no training data loaded."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def trained_client():
    """Test client pre-loaded with a small set of training cases.
    NOTE: requires a running Ollama instance with nomic-embed-text.
    Skip automatically if Ollama is unreachable.
    """
    with TestClient(app) as c:
        # Quick connectivity check
        r = c.get("/health")
        if r.status_code == 200 and not r.json().get("ollama_reachable", False):
            pytest.skip("Ollama not reachable — skipping trained_client fixture")

        cases = {
            "cases": [
                {
                    "question": "Where is my order?",
                    "answer": "Please provide your order ID and we will track it for you.",
                    "category": "seguimiento_pedido",
                    "priority": 1,
                },
                {
                    "question": "I need a refund",
                    "answer": "Refunds take 5-7 business days. Please share your order ID.",
                    "category": "reembolsos",
                    "priority": 2,
                },
                {
                    "question": "How do I pay with PayPal?",
                    "answer": "Select PayPal as your payment method at checkout.",
                    "category": "opciones_pago",
                    "priority": 3,
                },
            ],
            "use_gpu": False,
        }
        r = c.post("/train-support", json=cases)
        assert r.status_code == 200, f"Training failed: {r.text}"
        yield c
