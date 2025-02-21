# FastAPI LLM Server

A FastAPI server implementation for local LLM interactions using Ollama, supporting text generation and embeddings with real-time system monitoring.

## Project Structure

```
LLM/
├── server/
│   └── API/
│       ├── main.py          # FastAPI server implementation
│       ├── script.py        # Test client
│       └── README.md        # Documentation
└── client/
    └── llm-client/
        └── src/
            └── screens/
                └── ChatInterface.tsx  # React chat interface
```

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

### Web Interface (`ChatInterface.tsx`)

React-based chat interface with:

- Real-time message updates
- System resource monitoring
- Error handling
- Loading states
- Responsive design
- Material-UI components

## Quick Start

1. Install dependencies:
```bash
# Server dependencies
pip install fastapi requests pydantic psutil gputil uvicorn

# Client dependencies
cd client/llm-client
npm install
```

2. Install Ollama and required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

3. Start server:
```bash
cd server/API
python -m uvicorn main:app --host 0.0.0.0 --port 8002
```

4. Start client:
```bash
cd client/llm-client
npm start
```

5. Test API (optional):
```bash
python script.py
```

## API Endpoints

### GET /
Returns service status and model configuration.

### GET /system-info
Returns current system resource usage including CPU, memory, and GPU stats.

### POST /generate
Generates text using the LLM model.

Request body:
```json
{
    "text": "Your prompt here",
    "max_length": 50,
    "use_gpu": true
}
```

### POST /embeddings
Generates vector embeddings for given texts.

Request body:
```json
{
    "texts": ["text1", "text2"],
    "use_gpu": true
}
```

## Requirements

### Server
- Python 3.7+
- FastAPI
- Uvicorn
- Ollama
- PSUtil
- GPUtil
- GPU (optional, enabled by default)

### Client
- Node.js 14+
- React 17+
- Material-UI 5+
- TypeScript 4+

## Features

- GPU acceleration support
- System resource monitoring
- CORS enabled
- Comprehensive logging
- Error handling
- Real-time chat interface
- Streaming responses
- Resource usage visualization

## Environment Variables

### Server
- `LLM_MODEL`: Model name (default: "mistral")
- `EMBEDDING_MODEL`: Embedding model name (default: "nomic-embed-text")
- `DEFAULT_GPU`: GPU usage flag (default: true)

### Client
- `API_BASE_URL`: API endpoint (default: "http://localhost:8002")

## Error Handling

The application includes comprehensive error handling for:
- API connection issues
- Model loading failures
- Generation errors
- System resource limitations
- Invalid inputs

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your license here]