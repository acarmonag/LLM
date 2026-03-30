import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent vector store with MMR reranking support."""

    def __init__(self, persist_path: str = "vector_store.json"):
        self._lock = asyncio.Lock()
        self._persist_path = persist_path
        # Internal storage: list of dicts with keys: case_id, category, question, answer, created_at, embedding
        self._entries: list[dict] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    async def add_case(self, case: dict, embedding: list[float]) -> str:
        """Add a case with its embedding. Returns case_id."""
        case_id = str(uuid.uuid4())
        norm_embedding = normalize(np.array(embedding).reshape(1, -1))[0].tolist()
        entry = {
            "case_id": case_id,
            "category": case.get("category", ""),
            "question": case.get("question", ""),
            "answer": case.get("answer", ""),
            "priority": case.get("priority", 1),
            "created_at": datetime.utcnow().isoformat(),
            "embedding": norm_embedding,
        }
        async with self._lock:
            self._entries.append(entry)
        return case_id

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Top-K cosine similarity search. Returns list of {case_id, score, case metadata}."""
        if not self._entries:
            return []

        try:
            query_vec = normalize(np.array(query_embedding).reshape(1, -1))
            embeddings_matrix = np.array([e["embedding"] for e in self._entries])
            similarities = cosine_similarity(query_vec, embeddings_matrix)[0]

            top_k = min(top_k, len(self._entries))
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                entry = self._entries[idx]
                results.append({
                    "case_id": entry["case_id"],
                    "score": float(similarities[idx]),
                    "category": entry["category"],
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "priority": entry.get("priority", 1),
                    "created_at": entry["created_at"],
                })
            return results
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    async def mmr_rerank(
        self,
        candidates: list[dict],
        query_embedding: list[float],
        top_n: int = 3,
        lambda_: float = 0.7,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance reranking.
        score(d) = lambda * sim(d, query) - (1 - lambda) * max(sim(d, selected))
        """
        if not candidates:
            return []
        if len(candidates) <= top_n:
            return candidates

        try:
            query_vec = normalize(np.array(query_embedding).reshape(1, -1))[0]

            # Build embedding matrix for candidates from store
            candidate_embeddings = []
            for c in candidates:
                entry = self._get_entry_by_id(c["case_id"])
                if entry:
                    candidate_embeddings.append(np.array(entry["embedding"]))
                else:
                    # Fallback: use zero vector (should not happen normally)
                    candidate_embeddings.append(np.zeros_like(query_vec))

            candidate_embeddings = np.array(candidate_embeddings)

            # Similarity of each candidate to query
            query_sims = cosine_similarity(candidate_embeddings, query_vec.reshape(1, -1)).flatten()

            # Pairwise similarities between candidates
            pairwise_sims = cosine_similarity(candidate_embeddings)

            selected_indices = []
            remaining_indices = list(range(len(candidates)))

            for _ in range(top_n):
                best_score = -float("inf")
                best_idx = -1

                for idx in remaining_indices:
                    relevance = query_sims[idx]
                    if selected_indices:
                        max_redundancy = max(pairwise_sims[idx][s] for s in selected_indices)
                    else:
                        max_redundancy = 0.0

                    mmr_score = lambda_ * relevance - (1 - lambda_) * max_redundancy
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx >= 0:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)

            return [candidates[i] for i in selected_indices]
        except Exception as e:
            logger.error(f"Error during MMR reranking: {e}")
            return candidates[:top_n]

    def _get_entry_by_id(self, case_id: str) -> Optional[dict]:
        for entry in self._entries:
            if entry["case_id"] == case_id:
                return entry
        return None

    async def delete_case(self, case_id: str) -> bool:
        """Delete a case by case_id. Returns True if found and deleted."""
        async with self._lock:
            for i, entry in enumerate(self._entries):
                if entry["case_id"] == case_id:
                    self._entries.pop(i)
                    return True
        return False

    async def get_all_cases(self) -> list[dict]:
        """Return metadata for all cases (no embeddings)."""
        return [
            {
                "case_id": e["case_id"],
                "category": e["category"],
                "question": e["question"],
                "answer": e["answer"],
                "priority": e.get("priority", 1),
                "created_at": e["created_at"],
            }
            for e in self._entries
        ]

    async def save(self):
        """Persist store to JSON file."""
        async with self._lock:
            try:
                data = []
                for entry in self._entries:
                    data.append({
                        "case_id": entry["case_id"],
                        "category": entry["category"],
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "priority": entry.get("priority", 1),
                        "created_at": entry["created_at"],
                        "embedding": entry["embedding"],
                    })
                with open(self._persist_path, "w") as f:
                    json.dump(data, f)
                logger.info(f"Vector store saved: {len(data)} entries to {self._persist_path}")
            except Exception as e:
                logger.error(f"Error saving vector store: {e}")

    async def load(self):
        """Load store from JSON file if it exists."""
        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)
            async with self._lock:
                self._entries = data
            logger.info(f"Vector store loaded: {len(data)} entries from {self._persist_path}")
        except FileNotFoundError:
            logger.info(f"No existing vector store at {self._persist_path}, starting fresh")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
