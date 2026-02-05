from __future__ import annotations
from dataclasses import dataclass
import requests
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union
import json
import os

from interfaces.llm.messages import LlmMessage
JSONDict = Dict[str, Any]

@dataclass
class OpenAICompatChatClient:
    chat_url: str
    model_name: str = "llama"
    timeout_s: int = 120
    temperature: float = 0.0

    def _url(self, path: str) -> str:
        """
        Replace the path of chat_url with `path`.
        """
        parts = urlsplit(self.chat_url)
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    
    def _post_json(self, url: str, payload: JSONDict, *, stream: bool = False) -> requests.Response:
        r = requests.post(url, json=payload, timeout=self.timeout_s, stream=stream)
        if r.status_code != 200:
            raise RuntimeError(f"llama-server HTTP {r.status_code}: {r.text[:1000]}")
        return r

    def chat(self, system: str, user: str, max_tokens: int, temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        }
        r = requests.post(self.chat_url, json=payload, timeout=self.timeout_s)
        if r.status_code != 200:
            # show the server’s explanation (often “Loading model”)
            raise RuntimeError(f"llama-server HTTP {r.status_code}: {r.text[:1000]}")
        print("Here is r:", r)
        data = r.json()
        print(f"Here is the json: {data}")

        data = r.json()

        # DEBUG: dump message payload when enabled
        if os.getenv("LLM_DEBUG", "").strip() in {"1", "true", "True", "yes", "YES"}:
            try:
                msg = (data.get("choices") or [{}])[0].get("message")
            except Exception:
                msg = None
            print("LLM_DEBUG message:", msg)


        return (data["choices"][0]["message"]["content"] or "").strip()

    def chat_message(self, system: str, user: str, max_tokens: int, temperature: Optional[float] = None) -> LlmMessage:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        }
        r = requests.post(self.chat_url, json=payload, timeout=self.timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"llama-server HTTP {r.status_code}: {r.text[:1000]}")
        data = r.json()
        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else None
        if not message:
            return LlmMessage(role="assistant", content="", reasoning_content=None)
        return LlmMessage(
            role=message.get("role") or "assistant",
            content=message.get("content") or "",
            reasoning_content=message.get("reasoning_content"),
        )
    
    def chat_stream(self, system: str, user: str, max_tokens: int) -> Iterator[str]:
        """
        Yields incremental text chunks as they arrive (Server-Sent Events)
        """
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        }
        r = self._post_json(self.chat_url, payload, stream=True)

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()

            if not line.startswith("data:"):
                continue

            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choice = (event.get("choices") or[{}])[0]
            delta = choice.get("delta") or {}
            chunk = delta.get("content")

            if chunk is None:
                msg = choice.get("message") or {}
                chunk = msg.get("content")
            if chunk:
                yield chunk

    def json_schema_chat(self, system: str, user: str, max_tokens: int, schema: dict) -> Any:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_object",
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = self._post_json(self.chat_url, payload)
        content = (r.json()["choices"][0]["message"]["content"] or "").strip()
        return json.loads(content)
