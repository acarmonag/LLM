from pydantic import BaseModel
from typing import List, Optional

class SupportCase(BaseModel):
    question: str
    answer: str
    category: str
    priority: Optional[int] = 1

class SupportEmbeddingInput(BaseModel):
    cases: List[SupportCase]
    use_gpu: Optional[bool] = True
    
class SupportConfig(BaseModel):
    threshold: float = 0.75
    top_k: int = 5
    max_context_length: int = 10
    preprocess_text: bool = True
    min_confidence: str = "media"