from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import random
from text.header_extractor import build_edited_text, build_text_from_header_and_body, build_paragraphs_from_header_and_body

# I do not create DocxLoader objects
# They are injected into this pipeline so I only need to type check it.
if TYPE_CHECKING:
    from inout.docx_loader import DocxLoader
    from services.ged_service import GedService
    from services.llm_service import LlmService
    from services.explainability import ExplainabilityRecorder
    from inout.explainability_writer import ExplainabilityWriter
    from services.docx_output_service import DocxOutputService

@dataclass
class FeedbackPipeline:
    loader: "DocxLoader"
    ged: "GedService"
    llm: "LlmService"
    explain: "ExplainabilityRecorder"
    explain_writer: "ExplainabilityWriter"
    docx_out: "DocxOutputService"

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
            # Extract name, student number and essay title (metadata) from essay
            classified = self.llm.extract_metadata(" ".join(raw_paragraphs), explain=self.explain)
            self.explain.log("LLM - metadata extraction", "Extracted essay metadata via JSON task")
            if isinstance(classified, dict):
                self.explain.log_kv("LLM", classified)

            # Rewrite: keep header fields on their own lines, then join the body.
            edited_text, header, body_paragraphs = build_edited_text(raw_paragraphs, classified)
            if header:
                self.explain.log_kv("DOCX", header)
            self.explain.log("DOCX", f"Body paragraphs after header removal: {len(body_paragraphs)}")

            # Check for grammar errors (use processed body text)
            sentences = list(body_paragraphs)
            self.explain.log("GED", f"Split into {len(sentences)} sentences")
            ged_results = self.ged.score(sentences, batch_size=cfg.ged.batch_size, explain=self.explain)
            self.explain.log("GED", f"Total results: {len(ged_results)}")

            # Correct sentences where GED found grammar errors
            error_idxs = [i for i, r in enumerate(ged_results) if r.has_error]
            if error_idxs:
                self.explain.log("GED", f"Error sentence count: {len(error_idxs)}")
            max_corrections = max(0, int(cfg.run.max_llm_corrections))
            if max_corrections > 0 and error_idxs:
                seed = int(hashlib.md5(docx_path.name.encode("utf-8")).hexdigest()[:8], 16)
                rng = random.Random(seed)
                sample_count = min(max_corrections, len(error_idxs))
                sampled_idxs = sorted(rng.sample(error_idxs, sample_count))
                to_correct = [sentences[i] for i in sampled_idxs]
                corrected = self.llm.correct_sentences(to_correct, explain=self.explain)
                for idx, new_text in zip(sampled_idxs, corrected):
                    original = sentences[idx]
                self.explain.log("LLM - grammar correction", f"Corrected sentence {idx + 1}")
                self.explain.log("LLM - grammar correction", f"Original: {original}")
                self.explain.log("LLM - grammar correction", f"Corrected: {new_text}")
                    sentences[idx] = new_text
            else:
                self.explain.log("LLM - grammar correction", "No corrections requested or no error sentences found")

            edited_body_text = " ".join(s.strip() for s in body_paragraphs if s and s.strip())
            corrected_body_text = " ".join(s.strip() for s in sentences if s and s.strip())
            corrected_text = build_text_from_header_and_body(header, sentences)
            header_lines = build_paragraphs_from_header_and_body(header, [])[:3]

            # Feedback to be added once feedback has been initiated
            feedback_paragraphs = ["(Feedback not available yet.)"]

            # Build the word document to be returned to the student
            output_path = cfg.paths.output_docx_folder / f"{docx_path.stem}.docx"
            self.docx_out.build_report_with_header_and_body(
                output_path=output_path,
                original_paragraphs=raw_paragraphs,
                edited_text=edited_text,
                header_lines=header_lines,
                edited_body_text=edited_body_text,
                corrected_body_text=corrected_body_text,
                feedback_paragraphs=feedback_paragraphs,
                include_edited_text=include_edited_text_section,
            )
            self.explain.log("DOCX", f"Wrote output document: {output_path}")
        except Exception as exc:
            error = exc
            self.explain.log("ERROR", f"LLM JSON extraction failed: {type(exc).__name__}: {exc}")
        finally:
            lines = self.explain.finish_doc()
            self.explain_writer.write(docx_path, lines)

        if error is not None:
            raise error
