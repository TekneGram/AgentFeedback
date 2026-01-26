from __future__ import annotations

from typing import List

from nlp.llm.client import OpenAICompatChatClient

SYSTEM = (
    "You are a careful English writing assistant.\n"
    "Fix grammar and word choice errors but keep the original meaning.\n"
    "Return ONLY the corrected sentence. No explanations. No quotes.\n"
)


def correct_sentences(client: OpenAICompatChatClient, sentences: List[str], max_tokens: int) -> List[str]:
    out: List[str] = []
    for s in sentences:
        text = (s or "").strip()
        if not text:
            out.append(s)
            continue
        corrected = client.chat(system=SYSTEM, user=text, max_tokens=max_tokens)
        corrected = (corrected or "").strip()
        if not corrected:
            corrected = text
        out.append(corrected)
    return out
