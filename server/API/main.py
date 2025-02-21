from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import json
import psutil
import GPUtil
import requests

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Ollama-based LLM API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de datos
class QueryInput(BaseModel):
    text: str
    max_length: int = 50
    use_gpu: Optional[bool] = True

class EmbeddingInput(BaseModel):
    texts: List[str]
    use_gpu: Optional[bool] = True

# Configuración de modelos
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_GPU = False

logger.info(f"Usando modelo LLM: {LLM_MODEL}")
logger.info(f"Usando modelo de embeddings: {EMBEDDING_MODEL}")
logger.info(f"GPU enabled by default: {DEFAULT_GPU}")

def get_system_info():
    """Obtiene información del sistema."""
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
    return {
        "message": "API LLM con Ollama funcionando",
        "llm_model": LLM_MODEL,
        "gpu_enabled": DEFAULT_GPU
    }

@app.get("/system-info")
async def system_info():
    """Endpoint para obtener información del sistema."""
    return get_system_info()

@app.post("/generate")
async def generate_text(query: QueryInput):
    """Genera texto usando el modelo LLM de Ollama."""
    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": query.text
        }
        if query.use_gpu:
            payload["options"] = {"gpu": True}
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
        
        generated_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    response_data = json.loads(line.decode('utf-8'))
                    if "response" in response_data:
                        generated_text += response_data["response"]
                    if response_data.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse line: {line}")
                    continue
        
        system_info = get_system_info()
        return {
            "generated_text": generated_text.strip(),
            "system_info": system_info
        }
            
    except Exception as e:
        logger.error(f"Error en generación de texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
async def get_embeddings(input_data: EmbeddingInput):
    """Obtiene embeddings usando `nomic-embed-text` en Ollama."""
    try:
        embeddings_results = []
        for text in input_data.texts:
            payload = {
                "model": EMBEDDING_MODEL,
                "prompt": text
            }
            if input_data.use_gpu:
                payload["options"] = {"gpu": True}
            
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Request failed with status {response.status_code}: {response.text}")
                
            try:
                embedding_data = response.json()
                embeddings_results.append(embedding_data)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {response.text}")
                raise Exception(f"Invalid JSON response: {e}")

        system_info = get_system_info()
        return {
            "embeddings": embeddings_results,
            "system_info": system_info
        }
    except Exception as e:
        logger.error(f"Error obteniendo embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)