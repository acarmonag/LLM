FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck + init.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full API source (shared with local dev)
COPY ../API/ .

# Copy init script from Docker_api
COPY init.sh .
RUN chmod +x init.sh

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    OLLAMA_URL=http://ollama:11434 \
    ALLOWED_ORIGINS=http://localhost:5173 \
    SIMILARITY_THRESHOLD=0.75 \
    TOP_K=5 \
    MIN_CONFIDENCE=media \
    ADMIN_PASSWORD=admin123 \
    LLM_MODEL=mistral \
    EMBEDDING_MODEL=nomic-embed-text

EXPOSE 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8002/health || exit 1

CMD ["./init.sh"]
