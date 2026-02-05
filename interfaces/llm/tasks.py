from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GrammarCorrectionRequest:
    sentences: list[str]
    max_tokens: int


@dataclass(frozen=True)
class GrammarCorrectionResult:
    corrected_sentences: list[str]
    thinking: Optional[list[str]] = None


@dataclass(frozen=True)
class MetadataExtractionRequest:
    text: str
    max_tokens: int


@dataclass(frozen=True)
class MetadataExtractionResult:
    student_name: str
    student_number: str
    essay_title: str
    essay: str
