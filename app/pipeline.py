from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import random
import time

# Text restructuring utilities
from text.header_extractor import build_edited_text, build_text_from_header_and_body, build_paragraphs_from_header_and_body

# Interfaces and protocols
from interfaces.config.app_config import AppConfigShape
from interfaces.docx.loader import DocxLoader as DocxLoaderProtocol
from interfaces.docx.output import DocxOutput
from interfaces.pipeline.pipeline import Pipeline

# Terminal UI helpers
from utils.terminal_ui import stage, Color, type_print

# Type-only imports to avoid runtime circular dependencies
if TYPE_CHECKING:
    from services.ged_service import GedService
    from services.llm_service import LlmService
    from services.explainability import ExplainabilityRecorder
    from inout.explainability_writer import ExplainabilityWriter

@dataclass
class FeedbackPipeline(Pipeline):
    """
    End to end feedback pipeline for a single DOCX files
    Responsibilities:
    - Load text from DOCX
    - Run grammar error detection
    - Apply limited LLM corrections
    - Generate multiple type of writing feedback
    - Produce an annotated DOCX output
    - Record explainability and timing information
    """

    # Injected dependencies
    loader: DocxLoaderProtocol
    ged: "GedService"
    llm: "LlmService"
    explain: "ExplainabilityRecorder"
    explain_writer: "ExplainabilityWriter"
    docx_out: DocxOutput

    def run_on_file(self, docx_path: Path, cfg: AppConfigShape) -> None:
        """
        Run the full feedback pipeline on a single DOCX file.
        """
        # Start timing the pipeline
        start_time = time.perf_counter()

        # ---- LOAD INPUT -----
        type_print(f"Loading paragraphs from doc {docx_path}", color=Color.BLUE)

        # Load raw paragraphs from the DOCS file
        raw_paragraphs = self.loader.load_paragraphs(docx_path)

        # Decide whether to include the edited text section in the output
        include_edited_text_section = (
            cfg.run.include_edited_text_section_policy
            and len(raw_paragraphs) > 1
            and (raw_paragraphs[1] or "").strip() != ""
        )

        # Reset and initialize the explainability tracking
        self.explain.reset()
        self.explain.start_doc(docx_path, include_edited_text=include_edited_text_section)
        self.explain.log("DOCX", f"Loaded {len(raw_paragraphs)} paragraphs")

        error: Exception | None = None
        classified = None
        try:
            # ---- EXTRACT META DATA ----

            # Ask the LLM to extract essay metadata (title, genre, etc.)
            with stage("Extracting metadata", color=Color.CYAN):
                classified = self.llm.extract_metadata(" ".join(raw_paragraphs), explain=self.explain)
            self.explain.log("LLM", "Extracted essay metadata via JSON task")
            if isinstance(classified, dict):
                self.explain.log_kv("LLM", classified)

            # ---- EDIT TEXT ----
            # Remove header and normalize body text
            with stage("Reformatting original text", color=Color.RED):
                edited_text, header, body_paragraphs = build_edited_text(
                    raw_paragraphs, 
                    classified
                )
            
            # Log extracted header and body statistics
            if header:
                self.explain.log_kv("DOCX", header)
            self.explain.log(
                "DOCX", 
                f"Body paragraphs after header removal: {len(body_paragraphs)}"
            )

            # ---- GRAMMAR ERROR DETECTION -----
            with stage("Running grammar error detection", color=Color.RED):
                sentences = list(body_paragraphs)
                self.explain.log("GED", f"Split into {len(sentences)} sentences")

                # Score sentences using GED model
                ged_results = self.ged.score(
                    sentences, 
                    batch_size=cfg.ged.batch_size, 
                    explain=self.explain
                )

                self.explain.log("GED", f"Total results: {len(ged_results)}")

            # ---- GRAMMAR ERROR CORRECTION ----
            with stage("Running grammar error corrections.", color=Color.CYAN):
                # Identify errors with detected errors
                error_idxs = [i for i, r in enumerate(ged_results) if r.has_error]
                if error_idxs:
                    self.explain.log("GED", f"Error sentence count: {len(error_idxs)}")
                max_corrections = max(0, int(cfg.run.max_llm_corrections))

                # Apply a limited number of deterministic LLM corrections
                if max_corrections > 0 and error_idxs:
                    # Seed RNG by filename for reproducible sampling
                    seed = int(hashlib.md5(docx_path.name.encode("utf-8")).hexdigest()[:8], 16)
                    rng = random.Random(seed)

                    # Sample sentences to correct
                    sample_count = min(max_corrections, len(error_idxs))
                    sampled_idxs = sorted(rng.sample(error_idxs, sample_count))
                    to_correct = [sentences[i] for i in sampled_idxs]

                    # Ask LLM to correct selected sentences
                    corrected = self.llm.correct_sentences(
                        to_correct, 
                        explain=self.explain
                    )

                    # Replace sentences and log changes
                    for idx, new_text in zip(sampled_idxs, corrected):
                        original = sentences[idx]
                        self.explain.log("LLM", f"Corrected sentence {idx + 1}")
                        self.explain.log("LLM", f"Original: {original}")
                        self.explain.log("LLM", f"Corrected: {new_text}")
                        sentences[idx] = new_text
                else:
                    self.explain.log(
                        "LLM", 
                        "No corrections requested or no error sentences found"
                    )

            # ---- RECONSTRUCT TEXT ------
            edited_body_text = " ".join(
                s.strip() for s in body_paragraphs if s and s.strip()
            )
            corrected_body_text = " ".join(
                s.strip() for s in sentences if s and s.strip()
            )
            corrected_text = build_text_from_header_and_body(
                header, 
                sentences
            )

            # Extract header lines for DOCX display
            header_lines = build_paragraphs_from_header_and_body(
                header, 
                []
            )[:3]

            # ------- FEEDBACK -------

            # ---- Topic Sentence ----
            #with stage("Providing topic sentence feedback...", color=Color.CYAN):
            ts_feedback = self.llm.analyze_topic_sentence(
                edited_body_text, 
                self.explain
            )

            # Cause-effect feedback
            ce_feedback = self.llm.cause_effect_feedback(
                edited_body_text, 
                self.explain
            )

            # Compare-contrast feedback
            cc_feedback = self.llm.compare_contrast_feedback(
                edited_body_text, 
                self.explain
            )

            # Hedging feedback
            hedging_feedback = self.llm.hedging_feedback(
                edited_body_text, 
                self.explain
            )

            # Content feedback
            content_feedback = self.llm.content_feedback(
                edited_body_text, 
                self.explain
            )

            # Conclusion sentence feedback
            conclusion_feedback = self.llm.conclusion_feedback(
                edited_body_text, 
                self.explain
            )

            # Combine and summarize all feedback
            all_feedback = (
                ts_feedback + "\n" 
                + ce_feedback + "\n" 
                + cc_feedback + "\n" 
                + hedging_feedback + "\n" 
                + content_feedback + "\n" 
                + conclusion_feedback
            )
            summarized_feedback = self.llm.summarize_personalize_feedback(
                all_feedback, 
                self.explain
            )

            # Final feedback block to insert into DOCX
            feedback_paragraphs = [
                "(Feedback not available yet.)", 
                ts_feedback, 
                ce_feedback, 
                cc_feedback, 
                hedging_feedback, 
                content_feedback, 
                conclusion_feedback, 
                summarized_feedback]
            
            # ------- BUILD OUTPUT DOCX -------
            type_print("Building the word document...", color=Color.RED)
            
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
            
            self.explain.log(
                "DOCX", 
                f"Wrote output document: {output_path}"
            )

            type_print("Complete", color=Color.GREEN)
        
        except Exception as exc:
            # Capture and log any pipeline failure.
            error = exc
            self.explain.log(
                "ERROR", 
                f"LLM JSON extraction failed: {type(exc).__name__}: {exc}"
            )
        finally:
            # ---- FINALIZATION ----
            elapsed_s = time.perf_counter() - start_time
            self.explain.log("TIMING", f"Pipeline elapsed: {elapsed_s:.3f}s")

            # Finalize explainability log and persist it
            lines = self.explain.finish_doc()
            self.explain_writer.write(docx_path, lines)

            type_print(
                f"Elapsed time: {elapsed_s:.3f}s", 
                color=Color.BLUE
            )

        # Re-raise any exception after cleanup
        if error is not None:
            raise error
