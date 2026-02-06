from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import spacy

from nlp_kv_cache.llm.kv_client import KvCacheClient, KvCacheHandle
from nlp_kv_cache.tasks.topic_sentence_analysis_kv import analyze_topic_sentence as analyze_topic_sentence_kv
from nlp_kv_cache.tasks.cause_effect_feedback_kv import route_cause_effect_feedback
from nlp_kv_cache.tasks.compare_contrast_feedback_kv import route_compare_contrast_feedback
from nlp_kv_cache.tasks.hedging_feedback_kv import route_hedging_feedback
from nlp_kv_cache.tasks.content_feedback_kv import compare_paragraphs as content_compare_paragraphs

if TYPE_CHECKING:
    from services.explainability import ExplainabilityRecorder


@dataclass
class LlmServiceKV:
    kv_client: KvCacheClient
    model_family: str = "instruct"
    max_tokens_sentence: int = 128
    max_tokens_sentence_thinking: int = 1024
    _cache: KvCacheHandle | None = None

    def prepare_cache(self, paragraph: str, explain: "ExplainabilityRecorder | None" = None) -> KvCacheHandle:
        self._cache = self.kv_client.ingest_paragraph(paragraph)
        if explain is not None:
            size = len(paragraph or "")
            explain.log("LLM - kv cache", f"Cached paragraph length: {size}")
            explain.log("LLM - kv cache", f"Cache hash: {self._cache.paragraph_hash}")
        return self._cache

    def _require_cache(self) -> KvCacheHandle:
        if self._cache is None:
            raise RuntimeError("KV cache not prepared. Call prepare_cache() first.")
        return self._cache

    def extract_metadata(self, text: str, explain: "ExplainabilityRecorder | None" = None) -> Any:
        if explain is not None:
            explain.log("LLM - metadata extraction", f"JSON prompt length: {len(text or '')}")
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
                "essay": {"type": "string"},
            },
            "required": ["student_name", "student_number", "essay_title", "essay"],
        }
        out = self.kv_client.json_schema_chat_no_cache(
            system=system,
            extra=s,
            max_tokens=1024,
            schema=schema,
        )
        if explain is not None:
            if isinstance(out, dict):
                explain.log("LLM - metadata extraction", f"JSON keys: {', '.join(sorted(out.keys()))}")
            else:
                explain.log("LLM - metadata extraction", f"JSON type: {type(out).__name__}")
        return out

    def correct_sentences(self, sentences: list[str], explain: "ExplainabilityRecorder | None" = None) -> list[str]:
        if explain is not None:
            explain.log("LLM - grammar correction", f"Correction sentence count: {len(sentences)}")
        max_tokens = self.max_tokens_sentence_thinking if self.model_family == "thinking" else self.max_tokens_sentence
        out: list[str] = []
        for s in sentences:
            text = (s or "").strip()
            if not text:
                out.append(s)
                continue
            system = (
                "You are a careful English writing assistant.\n"
                "Fix grammar and word choice errors but keep the original meaning.\n"
                "Return ONLY the corrected sentence. No explanations. No quotes.\n"
            )
            message = self.kv_client.chat_no_cache(
                system=system,
                extra=text,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            final = (message or "").strip()
            if not final:
                final = text
            out.append(final)
        if explain is not None:
            explain.log("LLM - grammar correction", f"Correction output count: {len(out)}")
        return out

    def analyze_topic_sentence(self, edited_sentences: str, explain: "ExplainabilityRecorder | None" = None) -> Any:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(edited_sentences)
        sentences = [sent.text for sent in doc.sents]
        edited_sentences_minus_topic = " ".join(sentences[1:])
        learner_topic_sentence = sentences[0] if sentences else ""
        system_generate = (
            "You are a writer of English.\n"
            "You write plain English.\n"
            "Read a paragraph that is missing the topic sentence\n"
            "Then write a topic sentence that introduces the topic of the paragraph.\n"
            "Write only one concise topic sentence that does not contain too many specific details from the paragraph.\n"
            "No comments. No analysis. No trailing text. No JSON.\n"
        )
        suggested_topic_sentence = self.kv_client.chat_no_cache(
            system=system_generate,
            extra=f"Write a topic sentence for this paragraph:\n{edited_sentences_minus_topic}",
            max_tokens=1024,
            temperature=0.5,
        )
        if explain is not None:
            explain.log("LLM - topic sentence analysis", f"Generate suggested sentence: {suggested_topic_sentence}")
        cache = self._require_cache()
        feedback = analyze_topic_sentence_kv(
            self.kv_client,
            cache,
            edited_sentences,
            learner_topic_sentence,
            suggested_topic_sentence,
            max_tokens=1024,
        )
        if explain is not None:
            explain.log("LLM - topic sentence analysis", f"Provide feedback: {feedback}")
        return feedback

    def cause_effect_feedback(self, explain: "ExplainabilityRecorder | None" = None) -> str:
        cache = self._require_cache()
        feedback, count, examples = route_cause_effect_feedback(self.kv_client, cache, max_tokens=512)
        if explain is not None:
            explain.log("LLM - cause effect", f"Extracted examples: {count}")
            if examples:
                explain.log("LLM - cause effect", f"Examples: {'; '.join(examples)}")
            else:
                explain.log("LLM - cause effect", "Examples: none")
            branch = "suggest" if count == 0 else ("feedback" if count == 1 else "praise")
            explain.log("LLM - cause effect", f"Branch: {branch}")
            explain.log("LLM - cause effect", f"Feedback: {feedback}")
        return feedback

    def compare_contrast_feedback(self, explain: "ExplainabilityRecorder | None" = None) -> str:
        cache = self._require_cache()
        feedback, count, examples = route_compare_contrast_feedback(self.kv_client, cache, max_tokens=512)
        if explain is not None:
            explain.log("LLM - compare contrast", f"Extracted examples: {count}")
            if examples:
                explain.log("LLM - compare contrast", f"Examples: {'; '.join(examples)}")
            else:
                explain.log("LLM - compare contrast", "Examples: none")
            branch = "suggest" if count == 0 else ("feedback" if count == 1 else "praise")
            explain.log("LLM - compare contrast", f"Branch: {branch}")
            explain.log("LLM - compare contrast", f"Feedback: {feedback}")
        return feedback

    def hedging_feedback(self, explain: "ExplainabilityRecorder | None" = None) -> str:
        cache = self._require_cache()
        feedback, count, examples = route_hedging_feedback(self.kv_client, cache, max_tokens=512)
        if explain is not None:
            explain.log("LLM - hedging", f"Extracted examples: {count}")
            if examples:
                explain.log("LLM - hedging", f"Examples: {'; '.join(examples)}")
            else:
                explain.log("LLM - hedging", "Examples: none")
            branch = "suggest" if count == 0 else "praise"
            explain.log("LLM - hedging", f"Branch: {branch}")
            explain.log("LLM - hedging", f"Feedback: {feedback}")
        return feedback

    def content_feedback(self, explain: "ExplainabilityRecorder | None" = None) -> str:
        cache = self._require_cache()
        feedback = content_compare_paragraphs(self.kv_client, cache, max_tokens=512)
        if explain is not None:
            explain.log("LLM - content feedback", f"Feedback: {feedback}")
        return feedback
