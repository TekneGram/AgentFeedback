from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import spacy
import json

from nlp.llm.client import OpenAICompatChatClient
from nlp.llm.tasks.test_task import answer, stream_answer
from nlp.llm.tasks.metadata_extraction import extract_metadata
from nlp.llm.tasks.grammar_correction import correct_sentences as correct_grammar_sentences
from nlp.llm.tasks.paragraph_analysis import generate_topic_sentence, analyze_topic_sentence

if TYPE_CHECKING:
    from services.explainability import ExplainabilityRecorder

@dataclass
class LlmService:
    client: OpenAICompatChatClient
    model_family: str = "instruct"
    max_tokens_sentence: int = 128
    max_tokens_sentence_thinking: int = 1024

    def answer(self, sentence: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - answer", f"Answer prompt length: {len(sentence or '')}")
        out = answer(self.client, sentence, max_tokens=self.max_tokens_sentence)
        if explain is not None:
            explain.log("LLM - answer", f"Answer response length: {len(out or '')}")
        return out
    
    def stream_answer(self, sentence: str, explain: "ExplainabilityRecorder | None" = None) -> str:
        if explain is not None:
            explain.log("LLM - stream", f"Stream prompt length: {len(sentence or '')}")
        out = stream_answer(self.client, sentence, max_tokens=self.max_tokens_sentence)
        if explain is not None:
            explain.log("LLM - stream", f"Streamed {len(out)} chunks")
        return out
    
    def extract_metadata(self, text: str, explain: "ExplainabilityRecorder | None" = None) -> Any:
        if explain is not None:
            explain.log("LLM - metadata extraction", f"JSON prompt length: {len(text or '')}")
        out = extract_metadata(self.client, text, max_tokens=1024)
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
        suggested_topic_sentence = generate_topic_sentence(self.client, edited_sentences_minus_topic, max_tokens=1024, temperature=0.5)
        if explain is not None:
            explain.log("LLM - topic sentence analysis", f"Generate suggested sentence: {suggested_topic_sentence}")
        feedback = analyze_topic_sentence(self.client, edited_sentences, learner_topic_sentence, suggested_topic_sentence, max_tokens=1024)
        if explain is not None:
            explain.log("LLM - topic sentence analysis", f"Provide feedback: {feedback}")
        return feedback
