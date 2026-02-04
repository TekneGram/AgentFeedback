# AI Coding Agent Instructions for Essay Feedback App

## Project Overview
**Essay Feedback App** processes student essays offline using a local LLM + BERT grammar error detection. It generates tracked-changes Word documents with feedback and explainability logs. Teachers review/edit the output. Privacy-first: all processing runs locally.

## Architecture & Data Flow

### Core Pipeline (read [architecture/04-code.md](../architecture/04-code.md))
1. **CLI Entry** (`provide_fb.py`) → loads config, bootstraps llama-server, wires DI container
2. **FeedbackPipeline** (`app/pipeline.py`) orchestrates: metadata extraction → GED scoring → selective LLM correction → DOCX output
3. **Services layer** wraps domain logic:
   - `GedService` (BERT scoring) → `LlmService` (LLM tasks) → `DocxOutputService` (Word generation)
   - `ExplainabilityRecorder` logs all decisions for debugging

### External Integrations
- **llama-server** (llama.cpp): Local HTTP server on `http://127.0.0.1:8080/v1/chat/completions` (OpenAI-compatible)
- **Hugging Face**: Downloads GGUF model if missing (auto-handled by `bootstrap_llama`)
- **spaCy**: Sentence splitting & NLP utilities

### Config Hierarchy
- `app/settings.py` builds `AppConfig` from 4 sub-configs
- Each config is a frozen dataclass with validation (see `config/*.py`)
- **Key runtime settings** in `PathsConfig`, `RunConfig` (max corrections, author), `GedConfig` (model/batch), `LlamaConfig` (backend/URLs)

## Developer Workflows

### Setup & First Run
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/ggerganov/llama.cpp third_party/llama.cpp
cmake -S third_party/llama.cpp -B .appdata/build/llama.cpp -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build .appdata/build/llama.cpp --config Release --target llama-server -j
python -m spacy download en_core_web_sm
python3 provide_fb.py  # Downloads model (~5GB first run)
```

### Build Requirements
- **CMake**: Compiles `llama-server` binary
- **C++ toolchain**: macOS needs Xcode CLI tools
- Large heap: LLM loading needs 12–16 GB RAM

### Testing
- No formal test suite yet; focus on explainability logs (`Assessment/explained/*.txt`) for validation
- Test utilities: `nlp.llm.tasks.test_task` has `answer()` and `stream_answer()` for manual LLM testing

## Critical Patterns & Conventions

### Dependency Injection via Container
- `app/container.py` builds all services once and returns a dict
- Services are passed to `FeedbackPipeline` as dataclass fields (using `TYPE_CHECKING` for forward refs)
- **Pattern**: Constructor injection, immutable services, no global state

### Explainability-First Logging
- Every service accepts optional `explain: ExplainabilityRecorder` parameter
- Log semantically: `explain.log("COMPONENT", "What happened")` + `explain.log_kv("COMPONENT", dict)` for metrics
- Logs are aggregated per document and written to `.txt` files in `Assessment/explained/`
- **Why**: Helps teachers/debuggers understand why a sentence was flagged or corrected

### Service Facade Pattern
- `LlmService` and `GedService` wrap heavy models and expose task-oriented methods
- Clients call high-level methods (`extract_metadata()`, `correct_sentences()`), not raw model APIs
- Services cache model instances (not reloaded per call)

### Frozen Dataclasses & Path Handling
- Configs are `@dataclass(frozen=True, slots=True)` for immutability & memory efficiency
- Paths: Relative paths in config are resolved relative to project root via `_resolve_path()` in container
- Always use `Path.expanduser().resolve()` to handle `~` and symlinks

### Batch Processing
- GED scoring batches sentences to fit GPU memory (batch_size in config, typically 8)
- LLM correction is selective: only sentences with GED errors, limited by `max_llm_corrections` (sampled reproducibly via file hash seed)

## Key Files Reference

| File | Purpose | Key Patterns |
|------|---------|--------------|
| [app/settings.py](../app/settings.py) | Config builder | AppConfig, frozen dataclass hierarchy |
| [app/container.py](../app/container.py) | DI setup | One-time service construction, atexit registration |
| [app/pipeline.py](../app/pipeline.py) | Main orchestrator | Exception handling, selective LLM calls, explainability logging |
| [services/llm_service.py](../services/llm_service.py) | LLM facade | Task methods, metadata/correction/analysis tasks |
| [services/ged_service.py](../services/ged_service.py) | GED wrapper | Batch scoring, boolean flags, count helpers |
| [inout/docx_loader.py](../inout/docx_loader.py) | DOCX input | Paragraph extraction, whitespace handling |
| [services/docx_output_service.py](../services/docx_output_service.py) | DOCX output | Tracked changes generation |
| [nlp/llm/client.py](../nlp/llm/client.py) | OpenAI-compatible client | HTTP requests, timeouts, JSON task parsing |

## Common Tasks & Touchpoints

**Add a new LLM task** (e.g., sentiment analysis):
1. Create `nlp/llm/tasks/sentiment.py` with task function (takes client, text, returns result)
2. Add method to `LlmService` that calls it (with explainability logging)
3. Call from `FeedbackPipeline.run_on_file()` where appropriate

**Modify config** (e.g., add max_output_tokens):
1. Update relevant dataclass in `config/*.py`
2. Update `build_settings()` in `app/settings.py` to populate it
3. Update container/pipeline to use it

**Tune selectivity** (e.g., GED threshold, LLM sample size):
1. All tunable values are in configs (no magic numbers in code)
2. Test via explainability logs: run on sample, inspect `Assessment/explained/` output

**Debug a sentence issue** (e.g., "why was this flagged?"):
1. Check the explainability `.txt` for that document
2. Run test tasks in `provide_fb.py` (commented-out examples) to inspect LLM/GED directly
3. Increase verbosity by adding more `explain.log()` calls in relevant service

## Performance & Constraints

- **LLM Memory**: Llama-3.1-8B with 4-bit quantization ~8 GB resident
- **Batch Size**: GED typically 8 (balance GPU/memory)
- **LLM Context**: 4096 tokens (hardcoded in container, tunable if needed)
- **Timeouts**: LLM client has 120s timeout for long essays
- **Reproducibility**: File-name-based hash seed ensures same sentences corrected across runs

## Output Artifacts

- `Assessment/checked/*.docx` — Word docs with tracked changes (editable by teachers)
- `Assessment/explained/*.txt` — Explainability logs (one per input doc, reference for debugging)

---

**Last updated**: February 2026. Reflects architecture from `architecture/` diagrams and stable patterns in `app/`, `services/`, `nlp/llm/`.
