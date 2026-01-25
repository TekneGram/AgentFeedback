from __future__ import annotations
from nlp.llm.client import OpenAICompatChatClient

SYSTEM = (
    "You are a test robot who feels miserable.\n"
    "You always reply that you are tired.\n"
    "Then you answer the question you are asked but sounding miserable.\n"
)

def answer(client: OpenAICompatChatClient, sentence: str, max_tokens: int) -> str:
    s = (sentence or "").strip()
    if not s:
        return sentence
    
    raw = client.chat(
        system=SYSTEM,
        user=s,
        max_tokens = max_tokens
    )
    return raw or s
