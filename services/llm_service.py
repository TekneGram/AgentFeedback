from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from nlp.llm.client import OpenAICompatChatClient
from nlp.llm.tasks.test_task import answer, stream_answer
from nlp.llm.tasks.metadata_extraction import extract_metadata
from nlp.llm.tasks.grammar_correction import correct_sentences as correct_grammar_sentences

if TYPE_CHECKING:
    from services.explainability import ExplainabilityRecorder

@dataclass
class LlmService:
    client: OpenAICompatChatClient
    max_tokens_sentence: int = 128

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
        out = correct_grammar_sentences(self.client, sentences, max_tokens=self.max_tokens_sentence)
        if explain is not None:
            explain.log("LLM - grammar correction", f"Correction output count: {len(out)}")
        return out
