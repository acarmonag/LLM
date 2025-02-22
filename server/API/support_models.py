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
    max_context_length: int = 5
    preprocess_text: bool = True
    min_confidence: str = "media"