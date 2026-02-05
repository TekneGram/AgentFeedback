from __future__ import annotations

from typing import List

from interfaces.llm.client import LlmClient

SYSTEM = (
    "You are a careful English writing assistant.\n"
    "Fix grammar and word choice errors but keep the original meaning.\n"
    "Return ONLY the corrected sentence. No explanations. No quotes.\n"
)


def correct_sentences(client: LlmClient, sentences: List[str], max_tokens: int, *, model_family: str) -> List[tuple[str, str | None]]:
    out: List[tuple[str, str | None]] = []
    for s in sentences:
        text = (s or "").strip()
        if not text:
            out.append((s, None))
            continue
        message = client.chat_message(system=SYSTEM, user=text, max_tokens=max_tokens)
        thinking = (message.reasoning_content or "").strip() or None
        final = (message.content or "").strip()
        if not final and model_family == "thinking" and thinking:
            last_sentence = ""
            sentence = ""
            for ch in thinking:
                sentence += ch
                if ch in ".!?":
                    candidate = sentence.strip()
                    if candidate:
                        last_sentence = candidate
                    sentence = ""
            if not last_sentence:
                last_sentence = (sentence or thinking).strip()
            final = last_sentence
        if not final:
            final = text
        out.append((final, thinking))
    return out
