#!/bin/bash
set -e

echo "Starting initialization..."

check_ollama() {
    curl -sf "${OLLAMA_URL:-http://ollama:11434}/api/tags" >/dev/null 2>&1
}

pull_model() {
    local model=$1
    echo "Pulling model: $model..."
    curl -X POST "${OLLAMA_URL:-http://ollama:11434}/api/pull" \
         -H "Content-Type: application/json" \
         -d "{\"name\":\"$model\"}" \
         --no-progress-meter

    if curl -s "${OLLAMA_URL:-http://ollama:11434}/api/tags" | grep -q "$model"; then
        echo "Model $model pulled successfully"
        return 0
    else
        echo "Failed to pull model $model"
        return 1
    fi
}

echo "Waiting for Ollama to be ready..."
TIMEOUT=300
ELAPSED=0
while ! check_ollama; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Timeout waiting for Ollama after ${TIMEOUT}s"
        exit 1
    fi
    echo "Ollama not ready yet... (${ELAPSED}s elapsed)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo "Ollama is ready!"
pull_model "${LLM_MODEL:-mistral}" || exit 1
pull_model "${EMBEDDING_MODEL:-nomic-embed-text}" || exit 1

echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 8002
