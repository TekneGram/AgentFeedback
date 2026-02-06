from __future__ import annotations

from nlp_kv_cache.llm.kv_client import KvCacheClient, KvCacheHandle


SYSTEM_EXTRACT = (
    "Extract compare/contrast language from the paragraph.\n"
    "Include contrast words/phrases (e.g., \"in contrast\", \"however\", \"but\"), "
    "comparative and superlative adjectives/adverbs, and other comparison signals.\n"
    "Return ONLY valid JSON.\n"
    "Schema:\n"
    "{"
    "\"examples\": [\"<word or phrase>\", ...]"
    "}\n"
    "If there is no compare/contrast language, return:\n"
    "{\"examples\": []}\n"
)

SYSTEM_SUGGEST = (
    "You are a helpful writing tutor.\n"
    "Provide one sentence that adds a supporting detail using compare/contrast language.\n"
    "Be concise. Output only the sentence.\n"
)

SYSTEM_FEEDBACK = (
    "You are a helpful writing tutor.\n"
    "Praise the writer for using compare/contrast language, then suggest one additional "
    "supporting detail using compare/contrast language.\n"
    "Be concise. Output only plain text.\n"
)

SYSTEM_PRAISE = (
    "You are a helpful writing tutor.\n"
    "Praise the writer for using compare/contrast language in their paragraph.\n"
    "Be concise. Output only plain text.\n"
)


def extract_compare_contrast_language(
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


def compare_contrast_suggestor(
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
    return (suggestion or "").strip() or "Try adding a compare/contrast sentence to support your idea."


def compare_contrast_feedback(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    max_tokens: int,
) -> str:
    feedback = kv_client.chat_with_cache(
        handle=cache,
        system=SYSTEM_FEEDBACK,
        extra="",
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (feedback or "").strip() or "Good use of compare/contrast language. Consider adding one more supporting detail."


def compare_contrast_praise(
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
    return (praise or "").strip() or "Strong use of compare/contrast language."


def route_compare_contrast_feedback(
    kv_client: KvCacheClient,
    cache: KvCacheHandle,
    max_tokens: int,
) -> tuple[str, int, list[str]]:
    examples = extract_compare_contrast_language(kv_client, cache, max_tokens=max_tokens)
    count = len(examples)
    if count <= 0:
        return compare_contrast_suggestor(kv_client, cache, max_tokens=max_tokens), count, examples
    if count == 1:
        return compare_contrast_feedback(kv_client, cache, max_tokens=max_tokens), count, examples
    return compare_contrast_praise(kv_client, cache, max_tokens=max_tokens), count, examples
