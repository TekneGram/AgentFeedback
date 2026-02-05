from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GedSentenceResult:
    sentence: str
    has_error: bool
    score: Optional[float] = None
