from __future__ import annotations

from typing import Any, Optional

from nlp.llm.client import OpenAICompatChatClient

import json

SYSTEM_GENERATE = (
    "You are a writer of English.\n"
    "You write plain English.\n"
    "Read a paragraph that is missing the topic sentence\n"
    "Then write a topic sentence that introduces the topic of the paragraph.\n"
    "Write only one concise topic sentence that does not contain too many specific details from the paragraph.\n"
    "No comments. No analysis. No trailing text. No JSON.\n"
)

SYSTEM_ANALYZE = (
    "You receive JSON and output only text.\n"
    "Parse the JSON.\n"
    "Complete the task provided in the JSON in response to the learner_text in the JSON.\n"
    "Do not output JSON.\n"
    "Be concise.\n"
)

def generate_topic_sentence(client: OpenAICompatChatClient, text: str, max_tokens: int, temperature: Optional[float] = None) -> Any:
    """
    Accepts a body paragraph minus the first sentence.
    Suggests a topic sentence to match it.
    Exposes temperature control to allow user to vary suggestions.
    """
    s = (text or "").strip()
    if not s:
        return text
    
    instruction = "Write a topic sentence for this paragraph: \n" + s
    suggested = client.chat(system=SYSTEM_GENERATE, user=instruction, max_tokens=max_tokens, temperature=None if temperature is None else temperature)
    suggested = (suggested or "").strip()
    if not suggested:
        suggested = "No suggestion given!"
    return suggested

def analyze_topic_sentence(client: OpenAICompatChatClient, text: str, learner_topic_sentence: str, suggested_topic_sentence: str, max_tokens: int) -> Any:
    """
    Accepts a body paragraph with the topic sentence AND a suggested topic sentence.
    Compares the writer's topic sentence to the suggested topic sentence
    and provides feedback to the user on how to improve their writing.
    """
    s = (text or "").strip()
    if not s:
        return text
    user_json = {
            "learner_text": text,
            "learner_topic_sentence": learner_topic_sentence,
            "good_topic_sentence": suggested_topic_sentence,
            "task": "Determine whether learner_topic_sentence is too general, too specific, off topic, or just right. If too general, too specific or off topic, explain why and offer the good_topic_sentence as an alternative."
        }
    user = json.dumps(user_json, ensure_ascii=False)
    analysis = client.chat(system=SYSTEM_ANALYZE, user=user, max_tokens=max_tokens, temperature=0.0)
    analysis = (analysis or "").strip()
    if not analysis:
        analysis = "No analysis given!"
    return analysis