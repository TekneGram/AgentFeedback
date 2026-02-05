from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LlmMessage:
    role: str
    content: str
    reasoning_content: Optional[str] = None
