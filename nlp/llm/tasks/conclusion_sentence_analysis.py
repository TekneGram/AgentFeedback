from __future__ import annotations

from typing import Any

from interfaces.llm.client import LlmClient


SYSTEM = (
    "A conclusion sentence is the last sentence of this paragraph.\n"
    "Basics: It should summarize the main idea of the paragraph.\n"
    "Extras: It can reiterate some key points from the paragraph.\n"
    "Extras: It can also evaluate, predict the future or make a call to action.\n"
    "Comment on the learner's achievement of the basics and the extras.\n"
    "Write concisely.\n"
    "Output only plain text.\n"
)

def evaluate_conclusion(client: LlmClient, paragraph: str, max_tokens: int, temperature: float) -> str:
    s = (paragraph or "").strip()
    if not s:
        return paragraph
    suggestion = client.chat(system=SYSTEM, user=s, max_tokens=max_tokens, temperature=temperature)
    return (suggestion or "").strip() or "Nice job with the conclusion sentence!"
