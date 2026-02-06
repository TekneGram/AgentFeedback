from __future__ import annotations

from typing import Any
import json

from nlp_kv_cache.llm.kv_client import KvCacheClient, KvCacheHandle


SYSTEM_ANALYZE = (
    "You receive JSON and output only text.\n"
    "Parse the JSON.\n"
    "Complete the task provided in the JSON in response to the learner_text in the JSON.\n"
    "Do not output JSON.\n"
    "Be concise.\n"
)


def analyze_topic_sentence(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    learner_text: str,
    learner_topic_sentence: str,
    suggested_topic_sentence: str,
    max_tokens: int,
) -> Any:
    s = (learner_text or "").strip()
    if not s:
        return learner_text
    user_json = {
        "learner_text": learner_text,
        "learner_topic_sentence": learner_topic_sentence,
        "good_topic_sentence": suggested_topic_sentence,
        "task": (
            "Determine whether learner_topic_sentence is too general, too specific, off topic, "
            "or just right. If too general, too specific or off topic, explain why and offer "
            "the good_topic_sentence as an alternative."
        ),
    }
    user = json.dumps(user_json, ensure_ascii=False)
    analysis = kv_client.chat_with_cache(
        handle=cache,
        system=SYSTEM_ANALYZE,
        extra=user,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    analysis = (analysis or "").strip()
    if not analysis:
        analysis = "No analysis given!"
    return analysis
