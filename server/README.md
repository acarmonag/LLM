# FastAPI LLM Server

A FastAPI server implementation for local LLM interactions using Ollama, supporting text generation and embeddings.

## Core Components

### Main Server (`main.py`)

A FastAPI application exposing endpoints for:

- Root endpoint (`GET /`) - Service status
- System monitoring (`GET /system-info`) - Hardware usage stats
- Text generation (`POST /generate`) - LLM text generation
- Text embeddings (`POST /embeddings`) - Vector embeddings generation

Uses Mistral as default LLM and nomic-embed-text for embeddings.

### Test Client (`script.py`)

Python script to test all API endpoints featuring:

- Automated endpoint testing
- System info monitoring
- Formatted output display
- Error handling

## Quick Start

1. Install dependencies:
```bash
pip install fastapi requests pydantic psutil gputil uvicorn
```

2. Install Ollama and required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

3. Start server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8002
```

4. Test API:
```bash
python script.py
```

## Requirements

- Python 3.7+
- Ollama
- GPU (optional, enabled by default)

## Features

- GPU acceleration support
- System resource monitoring
- CORS enabled
- Comprehensive logging
- Error handling

