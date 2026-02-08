from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import logging

import spacy

from interfaces.llm.client import LlmClient
from nlp.llm.config_resolver import resolve_request_config
from nlp.llm.tasks.test_task import answer, stream_answer
from nlp.llm.tasks.metadata_extraction import extract_metadata
from nlp.llm.tasks.grammar_correction import correct_sentences as correct_grammar_sentences
from nlp.llm.tasks.topic_sentence_analysis import generate_topic_sentence, analyze_topic_sentence
from nlp.llm.tasks.cause_effect_feedback import route_cause_effect_feedback
from nlp.llm.tasks.compare_contrast_feedback import route_compare_contrast_feedback
from nlp.llm.tasks.hedging_feedback import route_hedging_feedback
from nlp.llm.tasks.content_feedback import compare_paragraphs, filter_feedback
from nlp.llm.tasks.conclusion_sentence_analysis import evaluate_conclusion
from nlp.llm.tasks.summarize_personalize import summarize_personalize_feedback

if TYPE_CHECKING:
    from services.explainability import ExplainabilityRecorder
    from app.settings import AppConfig

logger = logging.getLogger(__name__)

@dataclass
class LlmService:
    client: LlmClient
    model_family: str = "instruct"
    app_cfg: "AppConfig | None" = None

    def _require_cfg(self) -> "AppConfig":
        if self.app_cfg is None:
            raise ValueError("LlmService requires app_cfg for request configuration.")
        return self.app_cfg

    def answer(self, sentence: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - answer", f"Answer prompt length: {len(sentence or '')}")
        req = resolve_request_config("answer", self._require_cfg())
        logger.debug("LLM answer request: %s", req)
        out = answer(self.client, sentence, max_tokens=req.max_tokens, temperature=req.temperature)
        if explain is not None:
            explain.log("LLM - answer", f"Answer response length: {len(out or '')}")
        return out
    
    def stream_answer(self, sentence: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - stream", f"Stream prompt length: {len(sentence or '')}")
        req = resolve_request_config("stream_answer", self._require_cfg())
        logger.debug("LLM stream request: %s", req)
        out = stream_answer(self.client, sentence, max_tokens=req.max_tokens)
        if explain is not None:
            explain.log("LLM - stream", f"Streamed {len(out)} chunks")
        return out
    
    def extract_metadata(self, text: str, explain: "ExplainabilityRecorder | None" = None) -> Any:
        if explain is not None:
            explain.log("LLM - metadata extraction", f"JSON prompt length: {len(text or '')}")
        req = resolve_request_config("metadata_extraction", self._require_cfg())
        logger.debug("LLM metadata request: %s", req)
        out = extract_metadata(self.client, text, max_tokens=req.max_tokens)
        if explain is not None:
            if isinstance(out, dict):
                explain.log("LLM - metadata extraction", f"JSON keys: {', '.join(sorted(out.keys()))}")
            else:
                explain.log("LLM - metadata extraction", f"JSON type: {type(out).__name__}")
        return out

    def correct_sentences(self, sentences: list[str], explain: "ExplainabilityRecorder | None" = None) -> list[str]:
        if explain is not None:
            explain.log("LLM - grammar correction", f"Correction sentence count: {len(sentences)}")
        overrides = {"max_tokens": 1024} if self.model_family == "thinking" else None
        req = resolve_request_config("grammar_correction", self._require_cfg(), request_overrides=overrides)
        logger.debug("LLM grammar request: %s", req)
        max_tokens = req.max_tokens
        results = correct_grammar_sentences(
            self.client,
            sentences,
            max_tokens=max_tokens,
            model_family="thinking" if self.model_family == "thinking" else "instruct",
        )
        out: list[str] = []
        for idx, (final, thinking) in enumerate(results):
            out.append(final)
            if explain is not None and thinking:
                explain.log("LLM THINKING", f"Sentence {idx + 1}: {thinking}")
        if explain is not None:
            explain.log("LLM - grammar correction", f"Correction output count: {len(out)}")
        return out
    
    def analyze_topic_sentence(self, edited_sentences: str, explain: "ExplainabilityRecorder | None" = None) -> Any:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(edited_sentences)
        sentences = [sent.text for sent in doc.sents]
        edited_sentences_minus_topic = " ".join(sentences[1:])
        learner_topic_sentence = sentences[0]
        req_generate = resolve_request_config("topic_sentence_generate", self._require_cfg())
        logger.debug("LLM topic generate request: %s", req_generate)
        suggested_topic_sentence = generate_topic_sentence(
            self.client,
            edited_sentences_minus_topic,
            max_tokens=req_generate.max_tokens,
            temperature=req_generate.temperature,
        )
        if explain is not None:
            explain.log("LLM - topic sentence analysis", f"Generate suggested sentence: {suggested_topic_sentence}")
        req_analyze = resolve_request_config("topic_sentence_analyze", self._require_cfg())
        logger.debug("LLM topic analyze request: %s", req_analyze)
        feedback = analyze_topic_sentence(
            self.client,
            edited_sentences,
            learner_topic_sentence,
            suggested_topic_sentence,
            max_tokens=req_analyze.max_tokens,
            temperature=req_analyze.temperature,
        )
        if explain is not None:
            explain.log("LLM - topic sentence analysis", f"Provide feedback: {feedback}")
        return feedback

    def cause_effect_feedback(self, paragraph: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - cause effect", f"Paragraph length: {len(paragraph or '')}")
        req = resolve_request_config("cause_effect_feedback", self._require_cfg())
        logger.debug("LLM cause effect request: %s", req)
        feedback, count, examples = route_cause_effect_feedback(
            self.client, paragraph, max_tokens=req.max_tokens, temperature=req.temperature
        )
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

    def compare_contrast_feedback(self, paragraph: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - compare contrast", f"Paragraph length: {len(paragraph or '')}")
        req = resolve_request_config("compare_contrast_feedback", self._require_cfg())
        logger.debug("LLM compare contrast request: %s", req)
        feedback, count, examples = route_compare_contrast_feedback(
            self.client, paragraph, max_tokens=req.max_tokens, temperature=req.temperature
        )
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

    def hedging_feedback(self, paragraph: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - hedging", f"Paragraph length: {len(paragraph or '')}")
        req = resolve_request_config("hedging_feedback", self._require_cfg())
        logger.debug("LLM hedging request: %s", req)
        feedback, count, examples = route_hedging_feedback(
            self.client, paragraph, max_tokens=req.max_tokens, temperature=req.temperature
        )
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

    def content_feedback(self, paragraph: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - content feedback", f"Paragraph length: {len(paragraph or '')}")
        req_compare = resolve_request_config("content_compare", self._require_cfg())
        logger.debug("LLM content compare request: %s", req_compare)
        feedback = compare_paragraphs(
            self.client,
            paragraph,
            max_tokens=req_compare.max_tokens,
            temperature=req_compare.temperature,
        )
        if explain is not None:
            explain.log("LLM - content feedback", f"{feedback}")
        req_filter = resolve_request_config("content_filter", self._require_cfg())
        logger.debug("LLM content filter request: %s", req_filter)
        filtered_feedback = filter_feedback(
            self.client,
            feedback,
            max_tokens=req_filter.max_tokens,
            temperature=req_filter.temperature,
        )
        if explain is not None:
            explain.log("LLM - filtered content feedback", f"Feedback: {filtered_feedback}")
        return filtered_feedback
    
    def conclusion_feedback(self, paragraph: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        req = resolve_request_config("conclusion_feedback", self._require_cfg())
        logger.debug("LLM conclusion request: %s", req)
        feedback = evaluate_conclusion(
            self.client, paragraph, max_tokens=req.max_tokens, temperature=req.temperature
        )
        if explain is not None:
            explain.log("LLM - conclusion sentence feedback", f"{feedback}")
        return feedback
    
    def summarize_personalize_feedback(self, feedback: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        req = resolve_request_config("summarize_personalize", self._require_cfg())
        logger.debug("LLM summarize request: %s", req)
        summarized_feedback = summarize_personalize_feedback(
            self.client, feedback, max_tokens=req.max_tokens, temperature=req.temperature
        )
        if explain is not None:
            explain.log("LLM - summarize / personalize feedback", f"{summarized_feedback}")
        return summarized_feedback
