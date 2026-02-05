from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSelectionResult:
    model_key: str
    backend: str
    model_path: str
    display_name: str
