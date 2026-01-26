# provide_fb_personal_new.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import json
import re
import random
import hashlib

from docx import Document

from ged_bert_class import GedBertDetector
from llama_corrector import LlamaCorrector, LlamaConfig
from editing_class import TrackChangesEditor
from cause_effect_checker import CauseEffectChecker
from compare_contrast_checker import CompareContrastChecker

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DOCX_FOLDER = Path("../Assessment/in")
OUTPUT_DOCX_FOLDER = Path("../Assessment/checked")
EXPLAINED_TXT_FOLDER = Path("../Assessment/explained")

AUTHOR = "Daniel Parsons"

GED_MODEL_NAME = "gotutiyan/token-ged-bert-large-cased-bin"
GED_BATCH_SIZE = 8

LLAMA_BACKEND = "local"  # "local" or "server"
LLAMA_GGUF_PATH = "../Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLAMA_SERVER_MODEL = "llama"

SINGLE_PARAGRAPH_MODE = True

# Cap LLM corrections per doc
MAX_LLM_CORRECTIONS = 5


def extract_paragraph_texts(docx_path: Path) -> List[str]:
    doc = Document(str(docx_path))
    return [p.text or "" for p in doc.paragraphs]


# -----------------------------
# NEW: Line-end punctuation fixer
# -----------------------------
_END_PUNCT = {".", "!", "?"}
_TRAILING_CLOSERS = {'"', "”", "’", "'", ")", "]", "}", "»"}


def _ends_with_terminal_punct(s: str) -> bool:
    """
    True if s ends with .!? possibly followed by closing quotes/brackets.
    Examples treated as ending:
      'Hello.' , 'Hello."' , 'Hello.)'
    """
    t = (s or "").rstrip()
    if not t:
        return False

    # Strip trailing closers
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1].rstrip()

    return bool(t) and (t[-1] in _END_PUNCT)


def add_periods_to_line_ends(paragraphs: List[str]) -> List[str]:
    """
    Students sometimes put name/title/body on separate lines/paragraphs with no period.
    This ensures each LINE ends with a terminal punctuation (.), so sentence splitting
    treats them as separate sentences.

    We apply this to:
      - paragraph boundaries (because we later join paragraphs)
      - explicit line breaks inside a paragraph (\n from Shift+Enter)
    """
    out: List[str] = []

    for p in paragraphs:
        txt = (p or "").replace("\r\n", "\n")
        # Preserve explicit line breaks while we enforce punctuation per line
        lines = txt.split("\n")
        fixed_lines: List[str] = []

        for line in lines:
            s = line.strip()
            if not s:
                fixed_lines.append("")  # keep empty line
                continue

            if _ends_with_terminal_punct(s):
                fixed_lines.append(s)
            else:
                fixed_lines.append(s + ".")

        out.append("\n".join(fixed_lines))

    return out


def flatten_paragraphs_to_single(paragraphs: List[str]) -> str:
    parts: List[str] = []
    for p in paragraphs:
        # Turn explicit line breaks into spaces AFTER we've inserted periods where needed
        s = (p or "").replace("\r\n", "\n").replace("\n", " ").strip()
        if s:
            parts.append(s)
    merged = " ".join(parts).strip()
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


# -----------------------------
# JSON helpers for personalization (currently unused)
# -----------------------------
def serialize_feedback_json(paragraphs: List[str]) -> str:
    clean = [(p or "").replace("\r\n", "\n").rstrip() for p in paragraphs]
    payload = {"schema": "essaylens_feedback_v1", "paragraphs": clean}
    return json.dumps(payload, ensure_ascii=False)


def deserialize_feedback_json(s: str) -> List[str]:
    obj = json.loads(s)
    if not isinstance(obj, dict) or "paragraphs" not in obj or not isinstance(obj["paragraphs"], list):
        raise ValueError("Invalid feedback JSON returned from personalizer.")
    out: List[str] = []
    for p in obj["paragraphs"]:
        if p is None:
            out.append("")
        else:
            out.append(str(p).replace("\r\n", "\n").rstrip())
    return out


def personalize_one(llm: LlamaCorrector, text: str) -> Tuple[str, Optional[str]]:
    """
    Personalize one feedback paragraph (string) via llm.personalize_feedback(JSON).
    NOTE: unused right now; all calls remain commented out.
    """
    t = (text or "").strip()
    if not t:
        return "", None

    payload = serialize_feedback_json([t])
    try:
        out_json = llm.personalize_feedback(payload, expected_count=1)
        paras = deserialize_feedback_json(out_json)
        merged = "\n\n".join([p for p in paras if p is not None]).strip()
        return merged, None
    except Exception as e:
        return t, f"{type(e).__name__}: {e}"


def _stable_rng_for_file(filename: str) -> random.Random:
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    return random.Random(seed)


def prompt_teacher_start_sentence(file_name: str, edited_text: str, sentences: List[str]) -> int:
    n = len(sentences)
    print("\n" + "-" * 80)
    print(f"TEACHER REVIEW: {file_name}")
    print("-" * 80)
    print("Merged paragraph (EDITED TEXT):")
    print(edited_text if edited_text else "(empty)")
    print("\nSentences:")
    for i, s in enumerate(sentences, start=1):
        preview = (s or "").strip()
        if len(preview) > 200:
            preview = preview[:197] + "..."
        print(f"{i:>2}: {preview}")

    print("\nChoose the sentence number to START processing from.")
    print("Example: enter 2 to ignore sentence 1 (e.g., a title line).")

    while True:
        raw = input(f"Start from sentence [1-{n}] (default 1): ").strip()
        if raw == "":
            return 1
        try:
            k = int(raw)
            if k < 1:
                print("Please enter a number >= 1.")
                continue
            if k > n:
                print(f"Please enter a number <= {n}.")
                continue
            return k
        except ValueError:
            print("Please enter an integer (e.g., 1, 2, 3).")


def main() -> None:
    OUTPUT_DOCX_FOLDER.mkdir(parents=True, exist_ok=True)
    EXPLAINED_TXT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load GED once
    ged = GedBertDetector(model_name=GED_MODEL_NAME)

    # Load LLaMA once
    llama_cfg = LlamaConfig(
        backend=LLAMA_BACKEND,
        model_path=LLAMA_GGUF_PATH if LLAMA_BACKEND == "local" else "",
        server_url=LLAMA_SERVER_URL,
        server_model=LLAMA_SERVER_MODEL,
        temperature=0.0,
        max_tokens=128,
    )
    llm = LlamaCorrector(llama_cfg)

    editor = TrackChangesEditor(author=AUTHOR)
    ce_checker = CauseEffectChecker()
    cc_checker = CompareContrastChecker()

    docx_files = sorted(INPUT_DOCX_FOLDER.glob("*.docx"))
    if not docx_files:
        print(f"No .docx files found in: {INPUT_DOCX_FOLDER}")
        return

    run_stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for docx_path in docx_files:
        print("\n" + "=" * 80)
        print(f"FILE: {docx_path.name}")
        print("=" * 80)

        # IMPORTANT: keep raw original for ORIGINAL TEXT output
        original_paragraphs_raw = extract_paragraph_texts(docx_path)

        # Include EDITED TEXT only if:
        # - original has >1 paragraph AND
        # - second paragraph contains text
        include_edited_text_section = (
            len(original_paragraphs_raw) > 1 and (original_paragraphs_raw[1] or "").strip() != ""
        )

        # Teacher explainability lines (per file)
        teacher_lines: List[str] = []
        teacher_lines.append(f"Explainability Report: {docx_path.name}")
        teacher_lines.append(f"Generated (UTC): {run_stamp}")
        teacher_lines.append("")
        teacher_lines.append("=== RUN CONFIG ===")
        teacher_lines.append(f"AUTHOR: {AUTHOR}")
        teacher_lines.append(f"GED_MODEL: {GED_MODEL_NAME}")
        teacher_lines.append(f"GED_BATCH_SIZE: {GED_BATCH_SIZE}")
        teacher_lines.append(f"LLAMA_BACKEND: {LLAMA_BACKEND}")
        teacher_lines.append(f"SINGLE_PARAGRAPH_MODE: {SINGLE_PARAGRAPH_MODE}")
        teacher_lines.append(f"MAX_LLM_CORRECTIONS: {MAX_LLM_CORRECTIONS}")
        teacher_lines.append(f"INCLUDE_EDITED_TEXT_SECTION: {include_edited_text_section}")
        teacher_lines.append("")

        if not SINGLE_PARAGRAPH_MODE:
            raise RuntimeError("SINGLE_PARAGRAPH_MODE=False path not implemented in this script.")

        # -----------------------------
        # NEW: enforce sentence boundaries at line ends BEFORE flattening
        # -----------------------------
        analysis_paragraphs = add_periods_to_line_ends(original_paragraphs_raw)

        # 1) Force single paragraph ("EDITED TEXT" concept) from analysis paragraphs
        edited_text = flatten_paragraphs_to_single(analysis_paragraphs)

        teacher_lines.append("=== EDITED TEXT (forced single paragraph; after line-end punctuation fix) ===")
        teacher_lines.append(edited_text if edited_text else "(empty)")
        teacher_lines.append("")

        # 2) Sentence split on edited_text
        all_sentences = [s for s in editor.split_into_sentences(edited_text) if s.strip()]

        if not all_sentences:
            out_name = f"{docx_path.stem}_checked{docx_path.suffix}"
            out_path = OUTPUT_DOCX_FOLDER / out_name

            editor.build_single_paragraph_report(
                output_path=str(out_path),
                original_paragraphs=original_paragraphs_raw,  # RAW original output
                edited_text=edited_text,
                corrected_text=edited_text,
                feedback_heading="Language Feedback",
                feedback_paragraphs=["(No sentences detected in the document.)"],
                feedback_as_tracked_insertion=False,
                add_page_break_before_feedback=True,
                include_edited_text_section=include_edited_text_section,
            )

            teacher_lines.append("No sentences detected. No GED/LLM processing performed.")
            explain_path = EXPLAINED_TXT_FOLDER / f"{docx_path.stem}_explainability.txt"
            explain_path.write_text("\n".join(teacher_lines).rstrip() + "\n", encoding="utf-8")

            print(f"Saved (tracked changes): {out_path}")
            print(f"Saved (explainability): {explain_path}")
            continue

        # -----------------------------
        # Teacher chooses where processing starts (title/name handling)
        # -----------------------------
        start_from = prompt_teacher_start_sentence(docx_path.name, edited_text, all_sentences)
        ignored_sentences = all_sentences[: start_from - 1]
        process_sentences = all_sentences[start_from - 1 :]

        teacher_lines.append("=== TEACHER START-SENTENCE OVERRIDE ===")
        teacher_lines.append(f"Start processing from sentence: {start_from}")
        teacher_lines.append(f"Ignored leading sentences: {len(ignored_sentences)}")
        if ignored_sentences:
            teacher_lines.append("Ignored sentences (kept in output, not sent to GED/LLM/feedback):")
            for i, s in enumerate(ignored_sentences, start=1):
                teacher_lines.append(f"  - Sentence {i}: {s}")
        teacher_lines.append("")

        if not process_sentences:
            corrected_text = edited_text
            student_feedback_paragraphs = ["(No sentences selected for processing.)"]

            out_name = f"{docx_path.stem}_checked{docx_path.suffix}"
            out_path = OUTPUT_DOCX_FOLDER / out_name

            editor.build_single_paragraph_report(
                output_path=str(out_path),
                original_paragraphs=original_paragraphs_raw,
                edited_text=edited_text,
                corrected_text=corrected_text,
                feedback_heading="Language Feedback",
                feedback_paragraphs=student_feedback_paragraphs,
                feedback_as_tracked_insertion=False,
                add_page_break_before_feedback=True,
                include_edited_text_section=include_edited_text_section,
            )

            teacher_lines.append("No sentences selected for processing. No GED/LLM/feature feedback performed.")
            explain_path = EXPLAINED_TXT_FOLDER / f"{docx_path.stem}_explainability.txt"
            explain_path.write_text("\n".join(teacher_lines).rstrip() + "\n", encoding="utf-8")

            print(f"Saved (tracked changes): {out_path}")
            print(f"Saved (explainability): {explain_path}")
            continue

        # -----------------------------
        # 3) GED (batch) on PROCESS sentences only
        # -----------------------------
        ged_results = ged.score_sentences(process_sentences, batch_size=GED_BATCH_SIZE)
        error_flags_proc = [r.has_error for r in ged_results]
        num_errors_proc = sum(1 for f in error_flags_proc if f)

        print(f"Total sentences (all): {len(all_sentences)}")
        print(f"Processed sentences: {len(process_sentences)} (start_from={start_from})")
        print(f"Sentences flagged with errors (processed only): {num_errors_proc}")

        teacher_lines.append("=== GED RESULTS (processed sentences only) ===")
        teacher_lines.append(f"Total sentences (all): {len(all_sentences)}")
        teacher_lines.append(f"Processed sentences: {len(process_sentences)} (start_from={start_from})")
        teacher_lines.append(f"Flagged sentences (processed): {num_errors_proc}")
        teacher_lines.append("")

        # -----------------------------
        # 4) Choose up to MAX_LLM_CORRECTIONS flagged PROCESS sentences to correct
        # -----------------------------
        err_indices_proc = [i for i, has_err in enumerate(error_flags_proc) if has_err]

        if len(err_indices_proc) > MAX_LLM_CORRECTIONS:
            rng = _stable_rng_for_file(docx_path.name)
            chosen_err_indices_proc = set(rng.sample(err_indices_proc, MAX_LLM_CORRECTIONS))
        else:
            chosen_err_indices_proc = set(err_indices_proc)

        skipped_due_to_cap = len(err_indices_proc) - len(chosen_err_indices_proc)

        teacher_lines.append("=== LLM CORRECTION SELECTION (processed sentences only) ===")
        teacher_lines.append(f"Flagged sentences (processed): {len(err_indices_proc)}")
        teacher_lines.append(f"Selected for LLM correction: {len(chosen_err_indices_proc)}")
        teacher_lines.append(f"Skipped due to cap: {skipped_due_to_cap}")
        teacher_lines.append("")

        # -----------------------------
        # 5) Correct ONLY selected error sentences (LLM), on PROCESS sentences only
        # -----------------------------
        corrected_proc_sentences: List[str] = []
        corrections_made: List[Tuple[int, str, str]] = []  # (global_sent_idx_1based, before, after)

        for idx0_proc, (sent, has_err) in enumerate(zip(process_sentences, error_flags_proc)):
            global_idx1 = (len(ignored_sentences) + idx0_proc) + 1
            if has_err and (idx0_proc in chosen_err_indices_proc):
                fixed = llm.correct_one(sent)
                corrected_proc_sentences.append(fixed)
                if fixed.strip() != sent.strip():
                    corrections_made.append((global_idx1, sent, fixed))
            else:
                corrected_proc_sentences.append(sent)

        corrected_sentences_all = ignored_sentences + corrected_proc_sentences

        teacher_lines.append("=== GED/LLM STATUS BY SENTENCE (all sentences) ===")
        teacher_lines.append("")
        for idx0_all, sent in enumerate(all_sentences):
            idx1_all = idx0_all + 1
            if idx0_all < len(ignored_sentences):
                teacher_lines.append(f"[Sentence {idx1_all}]")
                teacher_lines.append("  Ignored by teacher override: True")
                teacher_lines.append("  GED: (skipped)")
                teacher_lines.append("  LLM called: False (skipped)")
                teacher_lines.append(f"  Text: {sent}")
                teacher_lines.append("")
                continue

            idx0_proc = idx0_all - len(ignored_sentences)
            has_err = error_flags_proc[idx0_proc]
            teacher_lines.append(f"[Sentence {idx1_all}]")
            teacher_lines.append("  Ignored by teacher override: False")
            teacher_lines.append(f"  GED flagged: {has_err}")
            teacher_lines.append(f"  Text: {sent}")

            if has_err and (idx0_proc in chosen_err_indices_proc):
                fixed = corrected_proc_sentences[idx0_proc]
                teacher_lines.append("  LLM called: True")
                teacher_lines.append(f"  Corrected: {fixed}")
                teacher_lines.append(f"  Changed: {fixed.strip() != sent.strip()}")
            elif has_err and (idx0_proc not in chosen_err_indices_proc):
                teacher_lines.append("  LLM called: False (skipped due to MAX_LLM_CORRECTIONS cap)")
            else:
                teacher_lines.append("  LLM called: False")

            teacher_lines.append("")

        teacher_lines.append("=== LLM GRAMMAR CORRECTIONS SUMMARY ===")
        teacher_lines.append(f"Corrections changed text: {len(corrections_made)}")
        for (idx1, before, after) in corrections_made:
            teacher_lines.append(f"- Sentence {idx1} changed")
            teacher_lines.append(f"  BEFORE: {before}")
            teacher_lines.append(f"  AFTER : {after}")
        teacher_lines.append("")

        # -----------------------------
        # 6) Reconstruct corrected single paragraph (FULL, with ignored prefix restored)
        # -----------------------------
        corrected_text = " ".join(corrected_sentences_all).strip()
        corrected_text = re.sub(r"\s+", " ", corrected_text).strip()

        teacher_lines.append("=== CORRECTED TEXT (single paragraph; ignored prefix restored) ===")
        teacher_lines.append(corrected_text if corrected_text else "(empty)")
        teacher_lines.append("")

        # -----------------------------
        # 7) Feature feedback on CORRECTED text, but ONLY from teacher-selected start sentence onward
        # -----------------------------
        feedback_text = " ".join(corrected_proc_sentences).strip()
        feedback_text = re.sub(r"\s+", " ", feedback_text).strip()

        teacher_lines.append("=== FEEDBACK TEXT (teacher-selected portion only) ===")
        teacher_lines.append(feedback_text if feedback_text else "(empty)")
        teacher_lines.append("")

        working_paragraphs = [feedback_text] if feedback_text else [""]

        student_feedback_paragraphs: List[str] = []
        teacher_lines.append("=== FEATURE FEEDBACK ===")
        teacher_lines.append("")

        # ---- Topic sentence ----
        teacher_lines.append("=== Topic Sentence Feedback ===")
        teacher_lines.append("Agent: LLM only (topic_sentence_feedback)")
        teacher_lines.append("")

        for p_idx, para in enumerate(working_paragraphs, start=1):
            fb = llm.topic_sentence_feedback(para) if para.strip() else ""
            # fb_personal, perr = personalize_one(llm, fb)  # KEEP COMMENTED OUT

            if fb.strip():
                student_feedback_paragraphs.append(fb.strip())

            teacher_lines.append(f"[Paragraph {p_idx}]")
            teacher_lines.append("Feature: Topic Sentence")
            teacher_lines.append("Detector: (none)")
            teacher_lines.append("LLM feedback (original):")
            teacher_lines.append(fb.strip() if fb.strip() else "(no feedback returned)")
            teacher_lines.append("")

        # ---- Cause–Effect ----
        teacher_lines.append("=== Cause–Effect Feedback ===")
        teacher_lines.append("Agent: CauseEffectChecker + LLM")
        teacher_lines.append("")

        for p_idx, para in enumerate(working_paragraphs, start=1):
            matches = ce_checker.find(para) if para.strip() else []
            occ = len(matches)
            used = ce_checker.phrases_used(para) if para.strip() else []

            fb = llm.cause_effect_feedback(para, phrases_used=used) if para.strip() else ""
            # fb_personal, perr = personalize_one(llm, fb)  # KEEP COMMENTED OUT

            if fb.strip():
                student_feedback_paragraphs.append(fb.strip())

            teacher_lines.append(f"[Paragraph {p_idx}]")
            teacher_lines.append("Feature: Cause–Effect")
            teacher_lines.append("Detector: cause_effect_checker")
            teacher_lines.append(f"Occurrences: {occ}")
            teacher_lines.append(f"Unique phrases: {', '.join(used) if used else '(none)'}")
            if matches:
                teacher_lines.append("Matches:")
                for m in matches:
                    surface = para[m.start:m.end]
                    teacher_lines.append(
                        f"  - surface='{surface}' lex='{m.phrase}' category='{m.category}' span=({m.start},{m.end})"
                    )
            else:
                teacher_lines.append("Matches: (none)")
            teacher_lines.append("LLM feedback (original):")
            teacher_lines.append(fb.strip() if fb.strip() else "(no feedback returned)")
            teacher_lines.append("")

        # ---- Compare–Contrast ----
        teacher_lines.append("=== Compare–Contrast Feedback ===")
        teacher_lines.append("Agent: CompareContrastChecker + LLM")
        teacher_lines.append("")

        for p_idx, para in enumerate(working_paragraphs, start=1):
            matches = cc_checker.find(para) if para.strip() else []
            occ = len(matches)
            used = cc_checker.phrases_used(para) if para.strip() else []

            fb = llm.compare_contrast_feedback(para, phrases_used=used) if para.strip() else ""
            # fb_personal, perr = personalize_one(llm, fb)  # KEEP COMMENTED OUT

            if fb.strip():
                student_feedback_paragraphs.append(fb.strip())

            teacher_lines.append(f"[Paragraph {p_idx}]")
            teacher_lines.append("Feature: Compare–Contrast")
            teacher_lines.append("Detector: compare_contrast_checker")
            teacher_lines.append(f"Occurrences: {occ}")
            teacher_lines.append(f"Unique expressions: {', '.join(used) if used else '(none)'}")
            if matches:
                teacher_lines.append("Matches:")
                for m in matches:
                    surface = para[m.start:m.end]
                    teacher_lines.append(f"  - surface='{surface}' category='{m.category}' span=({m.start},{m.end})")
            else:
                teacher_lines.append("Matches: (none)")
            teacher_lines.append("LLM feedback (original):")
            teacher_lines.append(fb.strip() if fb.strip() else "(no feedback returned)")
            teacher_lines.append("")

        # ---- Conclusion sentence ----
        teacher_lines.append("=== Conclusion Sentence Feedback ===")
        teacher_lines.append("Agent: LLM only (conclusion_sentence_feedback)")
        teacher_lines.append("")

        for p_idx, para in enumerate(working_paragraphs, start=1):
            fb = llm.conclusion_sentence_feedback(para) if para.strip() else ""
            # fb_personal, perr = personalize_one(llm, fb)  # KEEP COMMENTED OUT

            if fb.strip():
                student_feedback_paragraphs.append(fb.strip())

            teacher_lines.append(f"[Paragraph {p_idx}]")
            teacher_lines.append("Feature: Conclusion Sentence")
            teacher_lines.append("Detector: (none)")
            teacher_lines.append("LLM feedback (original):")
            teacher_lines.append(fb.strip() if fb.strip() else "(no feedback returned)")
            teacher_lines.append("")

        # ---- Final praise sentence ----
        teacher_lines.append("=== Final Praise Sentence ===")
        teacher_lines.append("Agent: LLM only (praise_sentence)")
        teacher_lines.append("")

        for p_idx, para in enumerate(working_paragraphs, start=1):
            fb = llm.praise_sentence(para) if para.strip() else ""
            if fb.strip():
                student_feedback_paragraphs.append(fb.strip())

            teacher_lines.append(f"[Paragraph {p_idx}]")
            teacher_lines.append("Feature: Praise sentence")
            teacher_lines.append("Detector: (none)")
            teacher_lines.append("LLM praise (original):")
            teacher_lines.append(fb.strip() if fb.strip() else "(no feedback returned)")
            teacher_lines.append("")

        # -----------------------------
        # 8) Output report doc (student-facing)
        # -----------------------------
        out_name = f"{docx_path.stem}_checked{docx_path.suffix}"
        out_path = OUTPUT_DOCX_FOLDER / out_name

        editor.build_single_paragraph_report(
            output_path=str(out_path),
            original_paragraphs=original_paragraphs_raw,  # RAW original output
            edited_text=edited_text,
            corrected_text=corrected_text,
            feedback_heading="Language Feedback",
            feedback_paragraphs=student_feedback_paragraphs,
            feedback_as_tracked_insertion=False,
            add_page_break_before_feedback=True,
            include_edited_text_section=include_edited_text_section,
        )

        # -----------------------------
        # 9) Output explainability txt (teacher-facing)
        # -----------------------------
        explain_path = EXPLAINED_TXT_FOLDER / f"{docx_path.stem}_explainability.txt"
        explain_path.write_text("\n".join(teacher_lines).rstrip() + "\n", encoding="utf-8")

        print(f"Saved (tracked changes): {out_path}")
        print(f"Saved (explainability): {explain_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
