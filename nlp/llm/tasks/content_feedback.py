from __future__ import annotations

from typing import Any

from interfaces.llm.client import LlmClient


SYSTEM_COMPARE = (
    "You are a reader who is interested in AI.\n"
    "You must compare two paragraphs about AI.\n"
    "Which paragraph is more engaging for the reader?\n"
    "Explain your choice by comparing:\n"
    "-Use of examples.\n"
    "-Clarity of main idea.\n"
    "-Reader interest and flow.\n"
    "Output only plain text.\n"
)

SYSTEM_FILTER = (
    "You are a feedback filter.\n"
    "You will see feedback about two paragraphs.\n"
    "Your job is to filter all references to the first paragraph\n"
    "You should keep only information a bout the second paragraph.\n"
    "Change all references to the second paragraph to just **the paragraph**.\n"
    "Do not use comparison language.\n"
    "Write in short sentences.\n"
    "Output only plain text.\n"
)

def compare_paragraphs(client: LlmClient, paragraph: str, max_tokens: int) -> str:
    s = (paragraph or "").strip()
    first_paragraph = "This is the first paragraph:\n\n AI is changing the world. I think AI will make us more useful in the future. Also, AI will help us to learn more things more quickly. AI is very interesting to use. But some people think AI is dangerous. I don't think so. Thank you."
    s = f"This is the learner's paragraph: {s}"
    s = first_paragraph + s
    if not s:
        return paragraph
    suggestion = client.chat(system=SYSTEM_COMPARE, user=s, max_tokens=max_tokens, temperature=0.2)
    return (suggestion or "").strip() or "Try adding a compare/contrast sentence to support your idea."

def filter_feedback(client: LlmClient, feedback: str, max_tokens: int) -> str:
    f = (feedback or "").strip()
    if not f:
        return feedback
    fb_new = client.chat(system=SYSTEM_FILTER, user=f, max_tokens=max_tokens, temperature=0.0)
    return (fb_new or "").strip() or "Nice work!"