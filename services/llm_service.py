from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from nlp.llm.client import OpenAICompatChatClient
from nlp.llm.tasks.test_task import answer, stream_answer, test_json

@dataclass
class LlmService:
    client: OpenAICompatChatClient
    max_tokens_sentence: int = 128

    def answer(self, sentence: str) -> str:
        
        out = answer(self.client, sentence, max_tokens=self.max_tokens_sentence)
        return out
    
    def stream_answer(self, sentence: str) -> str:
        out = stream_answer(self.client, sentence, max_tokens=self.max_tokens_sentence)
        return out
    
    def test_json(self, text: str) -> Any:
        out = test_json(self.client, text, max_tokens=1024)
        return out