from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import subprocess
import logging
import json
import psutil
import GPUtil

# Add near the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    use_gpu: bool = True

class EmbeddingInput(BaseModel):
    texts: List[str]
    use_gpu: bool = True

# Configuración de modelos
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_GPU = True

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

def run_ollama_command(command: str, use_gpu: bool = DEFAULT_GPU):
    """Ejecuta un comando de Ollama y devuelve la salida en JSON."""
    try:
        # Check if model exists
        model_name = command.split()[2]  # Get model name from command
        check_model = subprocess.run(
            f"ollama list | findstr {model_name}",
            shell=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8'  # Specify UTF-8 encoding
        )
        if check_model.returncode != 0:
            raise Exception(f"Model {model_name} not found. Please run 'ollama pull {model_name}'")
        
        if use_gpu:
            command += " --gpu"
        
        # Use UTF-8 encoding for subprocess
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            startupinfo=startupinfo
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
        return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Error ejecutando Ollama: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        command = f"ollama run {LLM_MODEL} '{query.text}'"
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            startupinfo=startupinfo
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
        
        system_info = get_system_info()
        return {
            "generated_text": result.stdout.strip(),
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
            # Create JSON payload
            payload = {
                "model": EMBEDDING_MODEL,
                "prompt": text
            }
            if input_data.use_gpu:
                payload["options"] = {"gpu": True}
            
            # Use PowerShell-compatible command
            json_escaped = json.dumps(payload).replace('"', '\\"')
            command = f'curl.exe -X POST http://localhost:11434/api/embeddings -H "Content-Type: application/json" -d "{json_escaped}"'
            
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                encoding='utf-8',  # Specify encoding explicitly
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Command failed: {result.stderr}")
                
            try:
                embedding_data = json.loads(result.stdout)
                embeddings_results.append(embedding_data)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON: {result.stdout}")
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