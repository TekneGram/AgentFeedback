from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from text.sentence_splitter import split_paragraphs

# I do not create DocxLoader objects
# They are injected into this pipeline so I only need to type check it.
if TYPE_CHECKING:
    from inout.docx_loader import DocxLoader
    from services.ged_service import GedService
    from services.llm_service import LlmService
    from services.explainability import ExplainabilityRecorder
    from inout.explainability_writer import ExplainabilityWriter

@dataclass
class FeedbackPipeline:
    loader: "DocxLoader"
    ged: "GedService"
    llm: "LlmService"
    explain: "ExplainabilityRecorder"
    explain_writer: "ExplainabilityWriter"

    def run_on_file(self, docx_path: Path, cfg) -> None:
        raw_paragraphs = self.loader.load_paragraphs(docx_path)
        include_edited_text_section = (
            cfg.run.include_edited_text_section_policy
            and len(raw_paragraphs) > 1
            and (raw_paragraphs[1] or "").strip() != ""
        )

        self.explain.reset()
        self.explain.start_doc(docx_path, include_edited_text=include_edited_text_section)
        self.explain.log("DOCX", f"Loaded {len(raw_paragraphs)} paragraphs")

        error: Exception | None = None
        classified = None
        try:
            classified = self.llm.test_json(" ".join(raw_paragraphs), explain=self.explain)
            self.explain.log("LLM", "Extracted essay metadata via JSON task")
            if isinstance(classified, dict):
                self.explain.log_kv("LLM", classified)

            sentences = split_paragraphs(raw_paragraphs)
            self.explain.log("GED", f"Split into {len(sentences)} sentences")
            ged_results = self.ged.score(sentences, batch_size=cfg.ged.batch_size, explain=self.explain)
            self.explain.log("GED", f"Total results: {len(ged_results)}")
        except Exception as exc:
            error = exc
            self.explain.log("ERROR", f"LLM JSON extraction failed: {type(exc).__name__}: {exc}")
        finally:
            lines = self.explain.finish_doc()
            self.explain_writer.write(docx_path, lines)

        if error is not None:
            raise error
