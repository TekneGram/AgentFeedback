from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from nlp.llm.client import OpenAICompatChatClient
from nlp.llm.tasks.test_task import answer

@dataclass
class LlmService:
    client: OpenAICompatChatClient
    max_tokens_sentence: int = 128

    def answer(self, sentence: str) -> str:
        
        out = answer(self.client, sentence, max_tokens=self.max_tokens_sentence)
        return out