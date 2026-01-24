from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nlp.ged_bert import GedBertDetector, GedSentenceResult

@dataclass(frozen=True, slots=True)
class GedService:
    """
    App-facing wrapper around GedBertDetector

    Keeps the detector (heavy model) alive and provides simple outputs
    that the pipeline needs (flags, counts, etc).
    """
    detector: GedBertDetector

    def score(self, sentences: List[str], batch_size: int) -> List[GedSentenceResult]:
        """
        Return full results (sentence + has_error).
        Useful for explainability.
        """
        if not sentences:
            return []
        
        return self.detector.score_sentences(sentences, batch_size=batch_size)
    
    def flag_sentences(self, sentences: List[str], batch_size: int) -> List[bool]:
        """
        Return only the boolean flags in the same order as input
        """
        return [r.has_error for r in self.score(sentences, batch_size=batch_size)]
    
    def count_flagged(self, sentences: List[str], batch_size: int) -> int:
        """
        Convenience helper
        """
        return sum(self.flag_sentences(sentences, batch_size=batch_size))