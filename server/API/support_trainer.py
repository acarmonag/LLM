import re
import logging

from support_models import SupportConfig
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class SupportTrainer:
    def __init__(self, config: SupportConfig = None, vector_store: VectorStore = None):
        if config is None:
            config = SupportConfig()
        self.threshold = config.threshold
        self.top_k = config.top_k
        self.vector_store = vector_store or VectorStore()
        # Keep a simple count accessible
        self._case_count = 0

    @property
    def cases(self):
        """Backward-compatible property for case count access."""
        return list(range(self._case_count))

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better comparison."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

    def add_cases(self, cases_with_embeddings):
        """Add cases with their embeddings to the vector store."""
        import asyncio
        loop = asyncio.get_event_loop()
        for case_data in cases_with_embeddings:
            try:
                case_data["case"]["question"] = self.preprocess_text(case_data["case"]["question"])
                # Use the vector store
                loop.run_until_complete(
                    self.vector_store.add_case(case_data["case"], case_data["embedding"])
                )
                self._case_count += 1
            except RuntimeError:
                # If there is already a running loop, schedule via create_task
                import asyncio
                asyncio.ensure_future(
                    self.vector_store.add_case(case_data["case"], case_data["embedding"])
                )
                self._case_count += 1
            except Exception as e:
                logger.error(f"Error adding case: {e}")
                continue

    async def add_cases_async(self, cases_with_embeddings):
        """Async version of add_cases."""
        for case_data in cases_with_embeddings:
            try:
                case_data["case"]["question"] = self.preprocess_text(case_data["case"]["question"])
                await self.vector_store.add_case(case_data["case"], case_data["embedding"])
                self._case_count += 1
            except Exception as e:
                logger.error(f"Error adding case: {e}")
                continue

    async def find_similar_cases_async(self, query_embedding, top_k=None):
        """Async two-stage retrieval: top-K search + MMR reranking."""
        if top_k is None:
            top_k = self.top_k

        if self.vector_store.size == 0:
            logger.warning("No embeddings available for similarity search")
            return []

        try:
            # Stage 1: top-K retrieval
            candidates = await self.vector_store.search(query_embedding, top_k=top_k)

            if not candidates:
                return []

            # Stage 2: MMR reranking to get top-3
            reranked = await self.vector_store.mmr_rerank(
                candidates, query_embedding, top_n=3, lambda_=0.7
            )

            # Format results for backward compat
            return [
                {
                    "case": {
                        "question": r["question"],
                        "answer": r["answer"],
                        "category": r["category"],
                        "priority": r.get("priority", 1),
                    },
                    "case_id": r["case_id"],
                    "similarity": r["score"],
                    "confidence": self._calculate_confidence(r["score"]),
                }
                for r in reranked
            ]
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []

    def find_similar_cases(self, query_embedding, top_k=None):
        """Sync wrapper for backward compat (used by /get-similar-cases endpoint)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # We are inside an async context, create a task
            import concurrent.futures
            # This should not be called from async context; use find_similar_cases_async instead
            # Fallback: direct numpy computation
            return self._find_similar_sync(query_embedding, top_k)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.find_similar_cases_async(query_embedding, top_k))
            finally:
                loop.close()

    def _find_similar_sync(self, query_embedding, top_k=None):
        """Direct sync computation as fallback."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize

        if top_k is None:
            top_k = self.top_k

        if self.vector_store.size == 0:
            return []

        try:
            entries = self.vector_store._entries
            query_vec = normalize(np.array(query_embedding).reshape(1, -1))
            embeddings_matrix = np.array([e["embedding"] for e in entries])
            similarities = cosine_similarity(query_vec, embeddings_matrix)[0]

            top_k_actual = min(top_k, len(entries))
            top_indices = np.argsort(similarities)[::-1][:top_k_actual]

            results = []
            for idx in top_indices:
                entry = entries[idx]
                results.append({
                    "case": {
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "category": entry["category"],
                        "priority": entry.get("priority", 1),
                    },
                    "case_id": entry["case_id"],
                    "similarity": float(similarities[idx]),
                    "confidence": self._calculate_confidence(float(similarities[idx])),
                })
            return results
        except Exception as e:
            logger.error(f"Error in sync similar cases: {e}")
            return []

    def _calculate_confidence(self, similarity: float) -> str:
        """Calculate confidence level based on similarity."""
        if similarity >= 0.85:
            return "alta"
        elif similarity >= self.threshold:
            return "media"
        else:
            return "baja"
