from __future__ import annotations
from interfaces.llm.client import LlmClient

SYSTEM = (
    "Always response in plain English. No JSON-looking text.\n"
)


def answer(client: LlmClient, sentence: str, max_tokens: int) -> str:
    s = (sentence or "").strip()
    if not s:
        return sentence
    raw = client.chat(
        system=SYSTEM,
        user=s,
        max_tokens=max_tokens,
    )
    return raw or s


def stream_answer(client: LlmClient, sentence: str, max_tokens: int) -> str:
    s = (sentence or "").strip()
    if not s:
        return sentence

    text = []
    for chunk in client.chat_stream(
        system="You are wonderfully witty! Always answer in plain English. No JSON-looking text.",
        user=s,
        max_tokens=max_tokens,
    ):
        print(chunk, end="", flush=True)
        text.append(chunk)

    return text
