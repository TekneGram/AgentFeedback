from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Optional

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import format_chatml
except Exception:  # pragma: no cover - optional dependency
    Llama = None
    format_chatml = None


_ASSISTANT_PREFIX = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class KvCacheHandle:
    paragraph_hash: str
    prefix_tokens: list[int]
    prefix_len: int
    stop: str | None


class KvCacheClient:
    """
    True in-memory KV-cache client using llama-cpp-python.
    Keeps one LLM instance alive and reuses KV without save/load.
    """

    _shared_llm: Llama | None = None

    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        thinking_mode: str = "no_think",
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
    ) -> None:
        if Llama is None or format_chatml is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it to use the KV-cache pipeline."
            )
        if KvCacheClient._shared_llm is None:
            params: dict[str, Any] = {
                "model_path": model_path,
                "n_ctx": n_ctx,
                "chat_format": "chatml",
            }
            if n_threads is not None:
                params["n_threads"] = n_threads
            if n_gpu_layers is not None:
                params["n_gpu_layers"] = n_gpu_layers
            KvCacheClient._shared_llm = Llama(**params)
        self._llm = KvCacheClient._shared_llm
        self._thinking_mode = thinking_mode
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._min_p = min_p
        self._presence_penalty = presence_penalty

    def _inject_thinking_mode(self, system: str) -> str:
        if self._thinking_mode == "no_think":
            return f"/no_think\n{system}"
        if self._thinking_mode == "think":
            return f"/think\n{system}"
        return system

    def _format_messages(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        res = format_chatml(messages)
        return res.prompt, res.stop

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True)

    def ingest_paragraph(self, paragraph: str) -> KvCacheHandle:
        p = (paragraph or "").strip()
        h = sha256(p.encode("utf-8")).hexdigest() if p else ""
        system = self._inject_thinking_mode(f"PARAGRAPH:\n{p}")
        prompt, stop = self._format_messages([{"role": "system", "content": system}])
        if prompt.endswith(_ASSISTANT_PREFIX):
            prompt = prompt[: -len(_ASSISTANT_PREFIX)]
        prefix_tokens = self._tokenize(prompt, add_bos=True)
        # Reset and eval prefix once to build KV cache in memory.
        self._llm.reset()
        if prefix_tokens:
            self._llm.eval(prefix_tokens)
        prefix_len = self._llm.n_tokens
        return KvCacheHandle(
            paragraph_hash=h,
            prefix_tokens=prefix_tokens,
            prefix_len=prefix_len,
            stop=stop,
        )

    def _build_user_prompt(self, system: str, extra: str) -> tuple[list[int], str | None]:
        content = system.strip()
        if extra:
            content = f"{content}\n\n{extra.strip()}"
        prompt, stop = self._format_messages([{"role": "user", "content": content}])
        tokens = self._tokenize(prompt, add_bos=False)
        return tokens, stop

    def chat_no_cache(
        self,
        system: str,
        extra: str,
        max_tokens: int,
        temperature: Optional[float] = None,
    ) -> str:
        system = self._inject_thinking_mode(system)
        prompt, stop = self._format_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": extra},
            ]
        )
        tokens = self._tokenize(prompt, add_bos=True)
        self._llm.reset()
        gen_tokens = self._generate_tokens(
            tokens,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
        )
        text = self._llm.detokenize(gen_tokens).decode("utf-8", errors="ignore")
        return text.strip()

    def chat_with_cache(
        self,
        handle: KvCacheHandle,
        system: str,
        extra: str,
        max_tokens: int,
        temperature: Optional[float] = None,
    ) -> str:
        # Rewind to prefix_len without clearing KV
        self._llm.n_tokens = handle.prefix_len
        user_tokens, stop = self._build_user_prompt(system, extra)
        gen_tokens = self._generate_tokens(
            user_tokens,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            reset=False,
        )
        text = self._llm.detokenize(gen_tokens).decode("utf-8", errors="ignore")
        return text.strip()

    def _generate_tokens(
        self,
        tokens: list[int],
        *,
        max_tokens: int,
        stop: str | None,
        temperature: Optional[float],
        reset: bool = False,
    ) -> list[int]:
        out: list[int] = []
        text = ""
        for token in self._llm.generate(
            tokens,
            reset=reset,
            top_p=self._top_p,
            top_k=self._top_k,
            min_p=self._min_p,
            temp=self._temperature if temperature is None else temperature,
            presence_penalty=self._presence_penalty,
        ):
            out.append(token)
            if stop:
                text = self._llm.detokenize(out).decode("utf-8", errors="ignore")
                if stop in text:
                    text = text.split(stop)[0]
                    return self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
            if len(out) >= max_tokens:
                break
        return out

    def json_schema_chat_no_cache(
        self,
        system: str,
        extra: str,
        max_tokens: int,
        schema: dict,
    ) -> Any:
        text = self.chat_no_cache(system=system, extra=extra, max_tokens=max_tokens, temperature=0.0)
        json_text = _extract_last_json_object(text)
        if json_text is None:
            raise ValueError(f"Invalid JSON from KV model: {text}")
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from KV model: {text}") from exc

    def json_schema_chat_with_cache(
        self,
        handle: KvCacheHandle,
        system: str,
        extra: str,
        max_tokens: int,
        schema: dict,
    ) -> Any:
        text = self.chat_with_cache(handle=handle, system=system, extra=extra, max_tokens=max_tokens, temperature=0.0)
        json_text = _extract_last_json_object(text)
        if json_text is None:
            raise ValueError(f"Invalid JSON from KV model: {text}")
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from KV model: {text}") from exc


def _extract_last_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder()
    last_json = None
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        raw = text[idx : idx + end].strip()
        if raw:
            last_json = raw
    return last_json
