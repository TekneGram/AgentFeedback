from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from nlp.llm.client import OpenAICompatChatClient
from nlp.llm.tasks.test_task import answer

@dataclass
class LlmService:
    client: OpenAICompatChatClient
    max_tokens_sentence: int = 128

    # Small, in-memory cache
    _cache: Dict[str, str] = None

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}

    def answer(self, sentence: str) -> str:
        key = sentence.strip()
        if not key:
            return sentence
        if key in self._cache:
            return self._cache(key)
        
        out = answer(self.client, key, max_tokens=self.max_tokens_sentence)
        self._cache[key] = out
        return out