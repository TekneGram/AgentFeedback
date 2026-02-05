from __future__ import annotations

from typing import Protocol, Optional, Any
from interfaces.llm.messages import LlmMessage


class LlmClient(Protocol):
    def chat(self, system: str, user: str, max_tokens: int, temperature: Optional[float] = None) -> str:
        ...

    def chat_message(self, system: str, user: str, max_tokens: int, temperature: Optional[float] = None) -> LlmMessage:
        ...

    def json_schema_chat(self, system: str, user: str, max_tokens: int, schema: dict) -> dict:
        ...
