import re
import logging
import json

import httpx
import numpy as np
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

ABBREVIATIONS = {
    "dont": "do not",
    "can't": "can not",
    "cant": "can not",
    "wont": "will not",
    "won't": "will not",
    "im": "i am",
    "i'm": "i am",
    "ive": "i have",
    "i've": "i have",
    "didnt": "did not",
    "didn't": "did not",
    "doesnt": "does not",
    "doesn't": "does not",
    "isnt": "is not",
    "isn't": "is not",
    "wasnt": "was not",
    "wasn't": "was not",
    "havent": "have not",
    "haven't": "have not",
    "hasnt": "has not",
    "hasn't": "has not",
    "wouldnt": "would not",
    "wouldn't": "would not",
    "couldnt": "could not",
    "couldn't": "could not",
    "shouldnt": "should not",
    "shouldn't": "should not",
    "thats": "that is",
    "that's": "that is",
    "whats": "what is",
    "what's": "what is",
    "heres": "here is",
    "here's": "here is",
    "theres": "there is",
    "there's": "there is",
    "youre": "you are",
    "you're": "you are",
    "theyre": "they are",
    "they're": "they are",
    "were": "we are",
    "we're": "we are",
}


class QueryProcessor:
    """Handles query preprocessing and expansion for improved RAG retrieval."""

    async def preprocess(self, query: str) -> str:
        """Lowercase, strip punctuation, expand abbreviations."""
        text = query.lower().strip()
        # Expand abbreviations
        for abbr, expansion in ABBREVIATIONS.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
        # Strip punctuation except for order IDs and emails
        # Preserve @ and . for emails, preserve ORD patterns
        text = re.sub(r'[^\w\s@.\-]', '', text)
        text = ' '.join(text.split())
        return text

    async def expand_query(self, query: str, ollama_url: str, http_client: httpx.AsyncClient) -> list[str]:
        """Call LLM to generate 3 alternative phrasings. Returns [original, alt1, alt2, alt3]."""
        try:
            prompt = f"Generate 3 alternative phrasings of this customer support query. Return only the phrasings, one per line, no numbering: {query}"
            payload = {
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
            }
            response = await http_client.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=30.0,
            )
            if response.status_code != 200:
                logger.warning(f"Query expansion LLM call failed: {response.status_code}")
                return [query]

            data = response.json()
            text = data.get("response", "")
            alternatives = [line.strip() for line in text.strip().split("\n") if line.strip()]
            # Take at most 3 alternatives
            alternatives = alternatives[:3]
            return [query] + alternatives
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    async def get_multi_embedding(
        self, queries: list[str], ollama_url: str, http_client: httpx.AsyncClient
    ) -> list[float]:
        """Embed all queries and return the averaged embedding."""
        embeddings = []
        for q in queries:
            try:
                payload = {
                    "model": "nomic-embed-text",
                    "prompt": q,
                }
                response = await http_client.post(
                    f"{ollama_url}/api/embeddings",
                    json=payload,
                    timeout=30.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings.append(np.array(data["embedding"]))
            except Exception as e:
                logger.warning(f"Failed to embed query '{q[:50]}...': {e}")
                continue

        if not embeddings:
            raise ValueError("Could not generate any embeddings for query expansion")

        avg_embedding = np.mean(embeddings, axis=0)
        normalized = normalize(avg_embedding.reshape(1, -1))[0]
        return normalized.tolist()
