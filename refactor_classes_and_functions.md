# Refactor plan: classes and function responsibilities

This file lists the proposed classes/modules, and which functions/methods belong where, based on your current codebase (`provide_fb.py` entrypoint + supporting modules).

---

## Folder structure (proposed)

```text
essaylens/
  provide_fb.py                  # thin CLI entrypoint (calls Pipeline)
  app/
    pipeline.py                  # orchestrates the whole run
    container.py                 # builds shared services (GED, LLM, editor, checkers)
    config.py                    # dataclasses for settings/paths
    models.py                    # dataclasses for internal results (DocRunResult, etc.)

  io/
    docx_loader.py               # read .docx -> paragraphs
    output_paths.py              # naming/output folder utilities
    explainability_writer.py     # writes teacher explainability .txt

  text/
    preprocessing.py             # add_periods_to_line_ends, flatten, etc.
    selection.py                 # teacher start sentence + stable sampling policy

  services/
    ged_service.py               # thin wrapper around GedBertDetector
    llm_service.py               # thin wrapper around LlamaCorrector
    feature_feedback.py          # calls CE/CC checkers + LLM for feedback strings

  features/
    cause_effect.py              # your cause_effect_checker.py
    compare_contrast.py          # your compare_contrast_checker.py

  docx/
    track_changes_editor.py      # your editing_class.py

  nlp/
    ged_bert.py                  # your ged_bert_class.py
    llama_corrector.py           # your llama_corrector.py

tests/
  test_preprocessing.py
  test_selection.py
  test_feature_detectors.py
```

---

## `app/config.py`

### `PathsConfig` (dataclass)
- `input_docx_folder`
- `output_docx_folder`
- `explained_txt_folder`

### `GedConfig` (dataclass)
- `model_name`
- `batch_size`

### `RunConfig` (dataclass)
- `author`
- `single_paragraph_mode`
- `max_llm_corrections`
- `include_edited_text_section_policy` *(optional)*

> Note: keep your existing `LlamaConfig` in `nlp/llama_corrector.py` or move it into `app/config.py` if you want all config in one place.

---

## `io/docx_loader.py`

### `DocxLoader`
**Responsibility:** isolate `.docx` reading from pipeline logic.

Methods:
- `load_paragraphs(docx_path) -> list[str]`  
  *(moves from `extract_paragraph_texts` in `provide_fb.py`)*

---

## `text/preprocessing.py` (utility module)

### Functions (pure utilities)
- `_ends_with_terminal_punct(s: str) -> bool`
- `add_periods_to_line_ends(paragraphs: list[str]) -> list[str]`
- `flatten_paragraphs_to_single(paragraphs: list[str]) -> str`

*(moves from punctuation/flatten helpers in `provide_fb.py`)*

---

## `text/selection.py`

### `TeacherSentenceChooser`
**Responsibility:** teacher-in-the-loop selection of starting sentence.

Methods:
- `choose_start_sentence(file_name: str, edited_text: str, sentences: list[str]) -> int`  
  *(moves from `prompt_teacher_start_sentence`)*

### `CorrectionSelector`
**Responsibility:** stable, repeatable sampling of which flagged sentences to correct (caps to N).

Methods:
- `select_indices(flagged_indices: list[int], max_corrections: int, stable_key: str) -> set[int]`
  - internally seeds RNG using `stable_key` (e.g., filename)

*(moves from `_stable_rng_for_file` + sampling logic in `provide_fb.py`)*

---

## `services/ged_service.py`

### `GedService`
**Responsibility:** one clean interface for grammar error detection in the app layer.

Constructor deps:
- `detector: GedBertDetector` (from `nlp/ged_bert.py`)

Methods:
- `flag_sentences(sentences: list[str], batch_size: int) -> list[bool]`
  - calls `GedBertDetector.score_sentences(...)`
  - converts scores to a boolean flag list

---

## `services/llm_service.py`

### `LlmService`
**Responsibility:** app-facing interface to your LLaMA corrector + feedback prompts.

Constructor deps:
- `client: LlamaCorrector` (from `nlp/llama_corrector.py`)

Methods:
- `correct_sentence(sentence: str) -> str`
- `topic_sentence_feedback(paragraph: str) -> str`
- `cause_effect_feedback(paragraph: str, phrases_used: list[str]) -> str`
- `compare_contrast_feedback(paragraph: str, phrases_used: list[str]) -> str`
- `conclusion_sentence_feedback(paragraph: str) -> str`
- `praise_sentence(paragraph: str) -> str`

---

## `services/feature_feedback.py`

### `FeatureFeedbackService`
**Responsibility:** detect rhetorical features and generate feedback text (student-facing + teacher explainability).

Constructor deps:
- `ce_checker: CauseEffectChecker`
- `cc_checker: CompareContrastChecker`
- `llm: LlmService`

Methods:
- `generate(paragraph: str) -> list[str]`
  - combines topic sentence feedback
  - CE detection → CE LLM feedback
  - CC detection → CC LLM feedback
  - conclusion feedback
  - praise
  - returns list of feedback paragraphs/blocks for the student report

- `explainability_blocks(paragraph: str) -> list[str]`
  - returns structured lines/blocks (for teacher log)

---

## `io/explainability_writer.py`

### `ExplainabilityWriter`
**Responsibility:** write teacher explainability logs as `.txt`.

Methods:
- `write(path, lines: list[str]) -> None`

### *(Optional)* `ExplainabilityLogBuilder`
**Responsibility:** makes teacher logs easier to build than manual `append`.

Methods:
- `add_section(title: str) -> None`
- `add_kv(key: str, value) -> None`
- `add_lines(lines: list[str]) -> None`
- `to_lines() -> list[str]`

---

## `app/pipeline.py`

### `FeedbackPipeline`
**Responsibility:** orchestrate the end-to-end run for one file (and optionally a folder).

Constructor deps (injected):
- `loader: DocxLoader`
- `editor: TrackChangesEditor`
- `ged: GedService`
- `llm: LlmService`
- `features: FeatureFeedbackService`
- `chooser: TeacherSentenceChooser`
- `selector: CorrectionSelector`
- `explain_writer: ExplainabilityWriter`

Methods:
- `run_on_file(docx_path, cfg: RunConfig) -> DocRunResult`
  - load raw paragraphs
  - preprocess paragraphs (punctuation + flattening)
  - split sentences
  - choose teacher starting sentence
  - GED flags sentences in processing window
  - select up to cap indices for correction
  - LLM correct only selected sentences
  - reconstruct corrected text
  - generate feature feedback for processing window
  - write docx report via `TrackChangesEditor.build_single_paragraph_report(...)`
  - write explainability `.txt`

- *(Optional)* `run_all(input_folder, cfg) -> list[DocRunResult]`

---

## Existing modules (kept, but moved into clearer packages)

### `docx/track_changes_editor.py`
- `TrackChangesEditor`
  - keep existing methods such as:
    - `split_into_sentences(...)`
    - `build_single_paragraph_report(...)`

### `nlp/ged_bert.py`
- `GedBertDetector`
  - keep existing:
    - `score_sentences(...)`

### `nlp/llama_corrector.py`
- `LlamaCorrector`
  - keep existing:
    - `correct_sentence(...)`
    - prompt-based feedback helpers (or call them through `LlmService`)

### `features/cause_effect.py`
- `CauseEffectChecker`
  - keep existing:
    - `find(...)`
    - `phrases_used(...)`
    - `count(...)`

### `features/compare_contrast.py`
- `CompareContrastChecker`
  - keep existing:
    - `find(...)`
    - `phrases_used(...)`
    - `count(...)`

---

## Entry point: `provide_fb.py` (after refactor)

Keep it thin:
- parse config / CLI args
- build container/services (`app/container.py`)
- call `FeedbackPipeline.run_all(...)` or loop calling `run_on_file(...)`
