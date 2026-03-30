import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

QUERY_LOG_FILE = os.getenv("QUERY_LOG_FILE", os.path.join(os.path.dirname(__file__), "query_log.json"))


@dataclass
class QueryLog:
    timestamp: str
    query: str
    matched_category: Optional[str]
    confidence: float
    response_time_ms: int
    session_id: Optional[str] = None
    rag_hit: bool = False


class AnalyticsStore:
    """Logs queries and provides aggregated analytics."""

    def __init__(self, path: str = QUERY_LOG_FILE):
        self._path = path
        self._logs: list[dict] = []
        self._load()

    def _load(self):
        try:
            with open(self._path, "r") as f:
                self._logs = json.load(f)
            logger.info(f"Analytics store loaded: {len(self._logs)} entries")
        except FileNotFoundError:
            self._logs = []
        except Exception as e:
            logger.error(f"Error loading analytics store: {e}")
            self._logs = []

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump(self._logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving analytics store: {e}")

    def log_query(self, log: QueryLog):
        self._logs.append(asdict(log))
        self._save()

    def get_stats(self, ollama_reachable: bool = False) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        today_logs = [e for e in self._logs if e.get("timestamp", "").startswith(today)]

        total_today = len(today_logs)

        rag_hits = [e for e in today_logs if e.get("rag_hit", False)]
        rag_hit_rate = round(len(rag_hits) / total_today * 100, 1) if total_today else 0.0

        confidences = [e["confidence"] for e in today_logs if "confidence" in e]
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        categories = [e["matched_category"] for e in today_logs if e.get("matched_category")]
        top_categories = [
            {"category": cat, "count": cnt}
            for cat, cnt in Counter(categories).most_common(5)
        ]

        return {
            "total_queries_today": total_today,
            "rag_hit_rate_pct": rag_hit_rate,
            "avg_confidence": avg_confidence,
            "top_categories": top_categories,
            "ollama_reachable": ollama_reachable,
        }
