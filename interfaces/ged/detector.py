from __future__ import annotations

from typing import Protocol
from interfaces.ged.results import GedSentenceResult


class GedDetector(Protocol):
    def score_sentences(self, sentences: list[str], batch_size: int = 8) -> list[GedSentenceResult]:
        ...
