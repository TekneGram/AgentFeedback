from __future__ import annotations

from typing import Any

from interfaces.llm.client import LlmClient


def extract_metadata(client: LlmClient, text: str, max_tokens: int) -> Any:
    s = (text or "").strip()
    if not s:
        return text
    system = (
        "Extract the student_name, student_number, essay_title, and essay.\n"
        "Do not edit any content you receive.\n"
        "Return ONLY valid JSON with double-quoted keys and string values.\n"
        "No extra text, no markdown, no trailing commas.\n"
        "Example:\n"
        "{"
        "\"student_name\":\"Daniel Parsons\","
        "\"student_number\":\"St29879.dfij9\","
        "\"essay_title\":\"Having Part Time Jobs\","
        "\"essay\":\"I disagree with...\""
        "}\n"
        "If there is no student_name leave the property blank.\n"
        "If there is no student_number leave the property blank.\n"
        "If there is no essay_title leave the property blank.\n"
        "Example:\n"
        "{"
        "\"student_name\":\"\","
        "\"student_number\":\"\","
        "\"essay_title\":\"\","
        "\"essay\":\"I disagree with...\""
        "}\n"
    )
    schema = {
        "type": "object",
        "properties": {
            "student_name": {"type": "string"},
            "student_number": {"type": "string"},
            "essay_title": {"type": "string"},
            "essay": {"type": "string"}
        },
        "required": ["student_name", "student_number", "essay_title", "essay"]
    }
    json = client.json_schema_chat(system, text, max_tokens=max_tokens, schema=schema)
    return json
