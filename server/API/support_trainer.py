import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re

class SupportTrainer:
    def __init__(self):
        self.cases = []
        self.embeddings = []
        self.threshold = 0.75  # Umbral de similitud configurable
        
    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto para mejorar la comparación."""
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales
        text = re.sub(r'[^\w\s]', '', text)
        # Normalizar espacios
        text = ' '.join(text.split())
        return text
    
    def add_cases(self, cases_with_embeddings):
        for case_data in cases_with_embeddings:
            # Preprocesar el texto de la pregunta
            case_data["case"]["question"] = self.preprocess_text(case_data["case"]["question"])
            self.cases.append(case_data["case"])
            # Normalizar el embedding
            embedding = np.array(case_data["embedding"])
            normalized_embedding = normalize(embedding.reshape(1, -1))[0]
            self.embeddings.append(normalized_embedding)
    
    def find_similar_cases(self, query_embedding, top_k=3):
        # Normalizar el embedding de la consulta
        query_embedding = normalize(np.array(query_embedding).reshape(1, -1))
        embeddings_array = np.array(self.embeddings)
        
        # Calcular similitudes
        similarities = cosine_similarity(query_embedding, embeddings_array)[0]
        
        # Filtrar por umbral
        valid_indices = np.where(similarities >= self.threshold)[0]
        if len(valid_indices) == 0:
            valid_indices = np.argsort(similarities)[-top_k:]
        
        # Ordenar por similitud
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:top_k]]
        
        return [
            {
                "case": self.cases[idx],
                "similarity": float(similarities[idx]),
                "confidence": self._calculate_confidence(similarities[idx])
            }
            for idx in top_indices
        ]
    
    def _calculate_confidence(self, similarity: float) -> str:
        """Calcula el nivel de confianza basado en la similitud."""
        if similarity >= 0.85:
            return "alta"
        elif similarity >= 0.75:
            return "media"
        else:
            return "baja"