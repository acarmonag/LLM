import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", os.path.join(os.path.dirname(__file__), "feedback.json"))


class FeedbackStore:
    """Persists user feedback ratings to a JSON file."""

    def __init__(self, path: str = FEEDBACK_FILE):
        self._path = path
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        try:
            with open(self._path, "r") as f:
                self._entries = json.load(f)
            logger.info(f"Feedback store loaded: {len(self._entries)} entries")
        except FileNotFoundError:
            self._entries = []
        except Exception as e:
            logger.error(f"Error loading feedback store: {e}")
            self._entries = []

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump(self._entries, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback store: {e}")

    def add_feedback(self, message_id: str, rating: int, comment: Optional[str] = None) -> dict:
        entry = {
            "message_id": message_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._entries.append(entry)
        self._save()

        if rating <= 2:
            logger.warning(
                f"Low-rated response (rating={rating}): message_id={message_id} comment={comment!r}"
            )

        return entry

    def get_stats(self) -> dict:
        if not self._entries:
            return {
                "total_count": 0,
                "average_rating": None,
                "low_rated_messages": [],
            }

        total = len(self._entries)
        avg = sum(e["rating"] for e in self._entries) / total
        low_rated = [
            e for e in self._entries if e["rating"] <= 2
        ]

        return {
            "total_count": total,
            "average_rating": round(avg, 2),
            "low_rated_messages": low_rated,
        }
