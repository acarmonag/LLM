from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import logging
import json
import psutil
import GPUtil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Ollama-based LLM API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class QueryInput(BaseModel):
    text: str
    max_length: int = 50
    use_gpu: bool = True

class EmbeddingInput(BaseModel):
    texts: List[str]
    use_gpu: bool = True

# Configuration from environment variables
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'ollama')
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
LLM_MODEL = os.getenv('LLM_MODEL', 'mistral')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
DEFAULT_GPU = os.getenv('DEFAULT_GPU', 'false').lower() == 'false'

logger.info(f"Using LLM model: {LLM_MODEL}")
logger.info(f"Using embeddings model: {EMBEDDING_MODEL}")
logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
logger.info(f"GPU enabled by default: {DEFAULT_GPU}")

def get_system_info():
    """Get system information."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                "id": gpu.id,
                "name": gpu.name,
                "load": f"{gpu.load*100}%",
                "memory_used": f"{gpu.memoryUsed}MB",
                "memory_total": f"{gpu.memoryTotal}MB",
                "temperature": f"{gpu.temperature}°C"
            })
    except:
        gpu_info = "No GPU information available"

    return {
        "cpu_usage": f"{cpu_percent}%",
        "memory_used": f"{memory.percent}%",
        "gpu_info": gpu_info
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "API LLM con Ollama funcionando",
        "llm_model": LLM_MODEL,
        "gpu_enabled": DEFAULT_GPU,
        "ollama_url": OLLAMA_BASE_URL
    }

@app.get("/system-info")
async def system_info():
    """System information endpoint."""
    return get_system_info()

@app.post("/generate")
async def generate_text(query: QueryInput):
    """Generate text using Ollama's API."""
    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": query.text,
            "options": {
                "gpu": query.use_gpu
            }
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        # Parse response and get only the generated text
        generated_text = response.json().get('response', '')
        
        system_info = get_system_info()
        return {
            "generated_text": generated_text,
            "system_info": system_info
        }
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
async def get_embeddings(input_data: EmbeddingInput):
    """Get embeddings using Ollama's API."""
    try:
        embeddings_results = []
        for text in input_data.texts:
            payload = {
                "model": EMBEDDING_MODEL,
                "prompt": text,
                "options": {
                    "gpu": input_data.use_gpu
                }
            }
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json=payload
            )
            response.raise_for_status()
            embeddings_results.append(response.json())

        system_info = get_system_info()
        return {
            "embeddings": embeddings_results,
            "system_info": system_info
        }
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)