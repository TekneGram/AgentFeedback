from __future__ import annotations
from dataclasses import dataclass
import requests

@dataclass
class OpenAICompatChatClient:
    chat_url: str
    model_name: str = "llama"
    timeout_s: int = 120
    temperature: float = 0.0

    def chat(self, system: str, user: str, max_tokens: int) -> str:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_token": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        }
        r = requests.post(self.chat_url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()