#!/bin/bash
set -e

echo "Starting initialization..."

# Function to check if Ollama is ready
check_ollama() {
    curl -sf "http://${OLLAMA_HOST}:${OLLAMA_PORT}" >/dev/null 2>&1
}

# Function to pull model with progress
pull_model() {
    local model=$1
    echo "Pulling model: $model..."
    curl -X POST "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/pull" \
         -H "Content-Type: application/json" \
         -d "{\"name\":\"$model\"}" \
         --no-progress-meter
    
    # Verify model was pulled
    if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" | grep -q "$model"; then
        echo "Model $model successfully pulled"
        return 0
    else
        echo "Failed to pull model $model"
        return 1
    fi
}

# Wait for Ollama with timeout
echo "Waiting for Ollama to be ready..."
TIMEOUT=300
ELAPSED=0
while ! check_ollama; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "Timeout waiting for Ollama after ${TIMEOUT}s"
        exit 1
    fi
    echo "Waiting for Ollama... (${ELAPSED}s)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

echo "Ollama is ready!"

# Pull models
pull_model "$LLM_MODEL" || exit 1
pull_model "$EMBEDDING_MODEL" || exit 1

echo "Starting FastAPI application..."
exec python -u main.py