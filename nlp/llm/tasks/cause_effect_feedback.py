from __future__ import annotations

from typing import Any

from interfaces.llm.client import LlmClient


SYSTEM_EXTRACT = (
    "Extract cause-effect phrases from the paragraph.\n"
    "Return ONLY valid JSON.\n"
    "Example paragraph:\n"
    "We have many problem in the world and this causes us to feel sad everyday. Therefore, we should...\n"
    "Schema:\n"
    "{"
    "\"examples\": [\"causes\", \"Therefore\", ...]"
    "}\n"
    "If there is no cause-effect language, return:\n"
    "{\"examples\": []}\n"
)

SYSTEM_SUGGEST = (
    "You are a helpful writing tutor.\n"
    "Provide one sentence that adds a supporting detail using cause-effect language.\n"
    "Be concise. Output only the sentence.\n"
)

SYSTEM_FEEDBACK = (
    "You are a helpful writing tutor.\n"
    "Praise the writer for using cause-effect language, then suggest one additional "
    "supporting detail using cause-effect language.\n"
    "Be concise. Output only plain text.\n"
)

SYSTEM_PRAISE = (
    "You are a helpful writing tutor.\n"
    "Praise the writer for using cause-effect language in their paragraph.\n"
    "Be concise. Output only plain text.\n"
)


def extract_cause_effect_language(client: LlmClient, paragraph: str, max_tokens: int) -> list[str]:
    s = (paragraph or "").strip()
    if not s:
        return []
    schema = {
        "type": "object",
        "properties": {
            "examples": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["examples"],
    }
    out = client.json_schema_chat(SYSTEM_EXTRACT, s, max_tokens=max_tokens, schema=schema)
    if isinstance(out, dict):
        examples = out.get("examples")
        if isinstance(examples, list):
            return [str(x).strip() for x in examples if str(x).strip()]
    return []


def cause_effect_suggestor(client: LlmClient, paragraph: str, max_tokens: int, temperature: float) -> str:
    s = (paragraph or "").strip()
    if not s:
        return paragraph
    suggestion = client.chat(system=SYSTEM_SUGGEST, user=s, max_tokens=max_tokens, temperature=temperature)
    return (suggestion or "").strip() or "Try adding a cause-effect sentence to support your idea."


def cause_effect_feedback(client: LlmClient, paragraph: str, max_tokens: int, temperature: float) -> str:
    s = (paragraph or "").strip()
    if not s:
        return paragraph
    feedback = client.chat(system=SYSTEM_FEEDBACK, user=s, max_tokens=max_tokens, temperature=temperature)
    return (feedback or "").strip() or "Good use of cause-effect language. Consider adding one more supporting detail."


def cause_effect_praise(client: LlmClient, paragraph: str, max_tokens: int, temperature: float) -> str:
    s = (paragraph or "").strip()
    if not s:
        return paragraph
    praise = client.chat(system=SYSTEM_PRAISE, user=s, max_tokens=max_tokens, temperature=temperature)
    return (praise or "").strip() or "Strong use of cause-effect language."


def route_cause_effect_feedback(
    client: LlmClient, paragraph: str, max_tokens: int, temperature: float
) -> tuple[str, int, list[str]]:
    examples = extract_cause_effect_language(client, paragraph, max_tokens=max_tokens)
    count = len(examples)
    if count <= 0:
        return cause_effect_suggestor(client, paragraph, max_tokens=max_tokens, temperature=temperature), count, examples
    if count == 1:
        return cause_effect_feedback(client, paragraph, max_tokens=max_tokens, temperature=temperature), count, examples
    return cause_effect_praise(client, paragraph, max_tokens=max_tokens, temperature=temperature), count, examples
