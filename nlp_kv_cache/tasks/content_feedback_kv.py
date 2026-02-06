from __future__ import annotations

from nlp_kv_cache.llm.kv_client import KvCacheClient, KvCacheHandle


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

FIRST_PARAGRAPH = (
    "This is the first paragraph:\n\n"
    "AI is changing the world. I think AI will make us more useful in the future. "
    "Also, AI will help us to learn more things more quickly. AI is very interesting to use. "
    "But some people think AI is dangerous. I don't think so. Thank you."
)


def compare_paragraphs(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    max_tokens: int,
) -> str:
    extra = f"{FIRST_PARAGRAPH}\n\nThis is the learner's paragraph:"
    suggestion = kv_client.chat_with_cache(
        handle=cache,
        system=SYSTEM_COMPARE,
        extra=extra,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (suggestion or "").strip() or "Try adding a compare/contrast sentence to support your idea."
