from __future__ import annotations

from nlp_kv_cache.llm.kv_client import KvCacheClient, KvCacheHandle


SYSTEM_EXTRACT = (
    "Extract language that shows uncertainty or certainty in the paragraph.\n"
    "Include hedging and strength-of-claim phrases (e.g., \"probably\", \"definitely\", "
    "\"maybe\", \"perhaps\", \"it could be said that\", \"should\", \"must\").\n"
    "Return ONLY valid JSON.\n"
    "Schema:\n"
    "{"
    "\"examples\": [\"<word or phrase>\", ...]"
    "}\n"
    "If there is no hedging or certainty language, return:\n"
    "{\"examples\": []}\n"
)

SYSTEM_SUGGEST = (
    "You are a helpful writing tutor.\n"
    "Provide one sentence that naturally adds hedging or strength-of-claim language.\n"
    "Be concise. Output only the sentence.\n"
)

SYSTEM_PRAISE = (
    "You are a helpful writing tutor.\n"
    "Praise the writer for using hedging or strength-of-claim language in their paragraph.\n"
    "Be concise. Output only plain text.\n"
)


def extract_hedging_language(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    max_tokens: int,
) -> list[str]:
    schema = {
        "type": "object",
        "properties": {
            "examples": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["examples"],
    }
    out = kv_client.json_schema_chat_with_cache(
        handle=cache,
        system=SYSTEM_EXTRACT,
        extra="",
        max_tokens=max_tokens,
        schema=schema,
    )
    if isinstance(out, dict):
        examples = out.get("examples")
        if isinstance(examples, list):
            return [str(x).strip() for x in examples if str(x).strip()]
    return []


def hedging_suggestor(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    max_tokens: int,
) -> str:
    suggestion = kv_client.chat_with_cache(
        handle=cache,
        system=SYSTEM_SUGGEST,
        extra="",
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (suggestion or "").strip() or "Try adding a sentence that shows uncertainty or strength of claim."


def hedging_praise(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    max_tokens: int,
) -> str:
    praise = kv_client.chat_with_cache(
        handle=cache,
        system=SYSTEM_PRAISE,
        extra="",
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (praise or "").strip() or "Good use of hedging or strength-of-claim language."


def route_hedging_feedback(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    max_tokens: int,
) -> tuple[str, int, list[str]]:
    examples = extract_hedging_language(kv_client, cache, max_tokens=max_tokens)
    count = len(examples)
    if count <= 0:
        return hedging_suggestor(kv_client, cache, max_tokens=max_tokens), count, examples
    return hedging_praise(kv_client, cache, max_tokens=max_tokens), count, examples
