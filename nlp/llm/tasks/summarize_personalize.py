from __future__ import annotations

from typing import Any

from interfaces.llm.client import LlmClient


SYSTEM = (
    "You are a friendly teacher who wants to support your student.\n"
    "There is too much feedback for the student.\n"
    "Take the main points of the feedback and summarize as follows:\n"
    "Focus on something done well.\n"
    "If something needs improving, explain it briefly here.\n"
    "End with other aspects that were done well.\n"
    "Ensure the feedback is logically organized and concise.\n"
    "Do not refer to 'the learner'. Speak directly to your student.\n"
)

def summarize_personalize_feedback(client: LlmClient, feedback: str, max_tokens: int) -> str:
    s = (feedback or "").strip()
    if not s:
        return feedback
    summary = client.chat(system=SYSTEM, user=s, max_tokens=max_tokens, temperature=0.2)
    return (summary or "").strip() or "Great work overall!"
