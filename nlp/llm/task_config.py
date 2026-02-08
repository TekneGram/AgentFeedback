from __future__ import annotations

from typing import Any, Dict

TASK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "answer": {"max_tokens": 128, "temperature": 0.0},
    "stream_answer": {"max_tokens": 128, "temperature": 0.0},
    "metadata_extraction": {"max_tokens": 1024, "temperature": 0.0},
    "grammar_correction": {"max_tokens": 128, "temperature": 0.0},
    "topic_sentence_generate": {"max_tokens": 1024, "temperature": 0.5},
    "topic_sentence_analyze": {"max_tokens": 1024, "temperature": 0.0},
    "cause_effect_feedback": {"max_tokens": 512, "temperature": 0.2},
    "compare_contrast_feedback": {"max_tokens": 512, "temperature": 0.2},
    "hedging_feedback": {"max_tokens": 512, "temperature": 0.2},
    "content_compare": {"max_tokens": 512, "temperature": 0.2},
    "content_filter": {"max_tokens": 256, "temperature": 0.0},
    "conclusion_feedback": {"max_tokens": 512, "temperature": 0.2},
    "summarize_personalize": {"max_tokens": 512, "temperature": 0.2},
}
