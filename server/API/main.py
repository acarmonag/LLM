from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import json
import psutil
import GPUtil
import requests
from support_models import SupportCase, SupportEmbeddingInput
from support_trainer import SupportTrainer
from simulated_orders import OrderDatabase
import re

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
    
class MessageContext(BaseModel):
    role: str
    content: str

class GenerateWithContext(BaseModel):
    text: str
    context: List[MessageContext] = []
    max_length: int = 50
    use_gpu: Optional[bool] = True

# Configuración de modelos
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_GPU = True

trainer = SupportTrainer()
order_db = OrderDatabase()

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
async def generate_text(query: GenerateWithContext):
    """Genera texto usando el modelo LLM de Ollama."""
    try:
        # Build context string from previous messages
        context_string = "\n".join([
            f"{msg.role}: {msg.content}" 
            for msg in query.context[-5:]  # Last 5 messages
        ])
        
        # Combine context with current query
        full_prompt = (
            f"{context_string}\n"
            f"Usuario: {query.text}\n"
            f"Asistente:"
        ) if context_string else f"Usuario: {query.text}\nAsistente:"

        payload = {
            "model": LLM_MODEL,
            "prompt": full_prompt
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
    
@app.post("/support-embeddings")
async def get_support_embeddings(input_data: SupportEmbeddingInput):
    """Obtiene embeddings para casos de soporte."""  
    try:
        embeddings_results = []
        for case in input_data.cases:
            combined_text = f"Q: {case.question}\nA: {case.answer}\nCategory: {case.category}"
            
            payload = {
                "model": EMBEDDING_MODEL,
                "prompt": combined_text
            }
            if input_data.use_gpu:
                payload["options"] = {"gpu": True}
            
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Request failed with status {response.status_code}")
                
            embedding_data = response.json()
            embeddings_results.append({
                "case": case.dict(),
                "embedding": embedding_data["embedding"]
            })

        return {
            "message": f"Generated embeddings for {len(embeddings_results)} support cases",
            "embeddings": embeddings_results
        }
    except Exception as e:
        logger.error(f"Error generating support embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-support")
async def train_support_system(input_data: SupportEmbeddingInput):
    """Entrena el sistema con nuevos casos de soporte."""
    embeddings_response = await get_support_embeddings(input_data)
    trainer.add_cases(embeddings_response["embeddings"])
    return {"message": "Training data added successfully"}

@app.post("/get-similar-cases")
async def find_similar_cases(query: QueryInput):
    """Encuentra casos similares basados en la consulta del usuario."""
    try:
        # Preprocesar la consulta
        processed_query = trainer.preprocess_text(query.text)
        
        # Verificar si hay un número de orden en la consulta
        order_id_match = re.search(r'ORD\d{6}', query.text)
        order_info = None
        
        if order_id_match:
            order_id = order_id_match.group()
            order_info = order_db.get_order(order_id)
            if order_info:
                # Personalizar la consulta con la información de la orden
                processed_query = f"{processed_query} Orden: {order_id} Estado: {order_info['status']}"
        
        # Obtener embedding de la consulta procesada
        embedding_input = EmbeddingInput(texts=[processed_query])
        embedding_result = await get_embeddings(embedding_input)
        
        # Buscar casos similares
        similar_cases = trainer.find_similar_cases(
            embedding_result["embeddings"][0]["embedding"]
        )
        
        # Enriquecer las respuestas con información de la orden si está disponible
        if order_info:
            for case in similar_cases:
                if "order_status" in case["case"]["category"]:
                    status_details = order_db._get_status_details(order_info)
                    case["case"]["answer"] = case["case"]["answer"].replace(
                        "[Detalles específicos serán insertados dinámicamente]",
                        f"""
                            Estado actual: {order_info['status']}
                            Fecha de orden: {status_details['order_date']}
                            Total: ${status_details['total']:.2f}
                            {"Número de seguimiento: " + status_details.get('tracking_number', 'No disponible') if order_info['status'] in ['Enviado', 'Entregado'] else ''}
                            {"Fecha de envío: " + status_details.get('shipping_date', '') if 'shipping_date' in status_details else ''}
                            {"Fecha de entrega: " + status_details.get('delivery_date', '') if 'delivery_date' in status_details else ''}
                            {"Motivo: " + status_details.get('decline_reason', '') if 'decline_reason' in status_details else ''}
                            {"Fecha de reembolso: " + status_details.get('refund_date', '') if 'refund_date' in status_details else ''}
                        """
                    )
        
        # Verificar correo electrónico en la consulta
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', query.text)
        if email_match:
            email = email_match.group()
            customer_orders = order_db.get_customer_orders(email)
            if customer_orders:
                orders_summary = "\n".join([
                    f"- Orden {order['order_id']}: {order['status']}"
                    for order in customer_orders
                ])
                similar_cases[0]["case"]["answer"] += f"\n\nÓrdenes encontradas:\n{orders_summary}"
        
        # Agregar metadatos adicionales
        response = {
            "similar_cases": similar_cases,
            "query_processed": processed_query,
            "threshold_used": trainer.threshold,
            "total_cases": len(trainer.cases),
            "order_info": order_info if order_info else None,
            "confidence_level": similar_cases[0]["confidence"] if similar_cases else "baja"
        }
        
        return response
    except Exception as e:
        logger.error(f"Error finding similar cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)