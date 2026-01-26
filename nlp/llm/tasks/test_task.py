from __future__ import annotations
from nlp.llm.client import OpenAICompatChatClient
from typing import Any

SYSTEM = (
    "Always response in plain English. No JSON-looking text.\n"
)

def answer(client: OpenAICompatChatClient, sentence: str, max_tokens: int) -> str:
    s = (sentence or "").strip()
    if not s:
        return sentence
    
    raw = client.chat(
        system=SYSTEM,
        user=s,
        max_tokens = max_tokens
    )
    return raw or s

def stream_answer(client: OpenAICompatChatClient, sentence: str, max_tokens: int) -> str:
    s = (sentence or "").strip()
    if not s:
        return sentence
    
    text = []
    for chunk in client.chat_stream(
        system="You are wonderfully witty! Always answer in plain English. No JSON-looking text.",
        user=s,
        max_tokens = max_tokens
    ):
        print(chunk, end="", flush=True)
        text.append(chunk)

    return text

def test_json(client: OpenAICompatChatClient, text: str, max_tokens: int) -> Any:
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
