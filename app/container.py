
# Import I/O, NLP and service-layer components.
from inout.docx_loader import DocxLoader
from inout.explainability_writer import ExplainabilityWriter
from nlp.ged_bert import GedBertDetector
from services.ged_service import GedService
from services.llm_service import LlmService
from services.explainability import ExplainabilityRecorder
from services.docx_output_service import DocxOutputService

# Import Llama server process + OpenAI-compatible client
from nlp.llm.server_process import LlamaServerProcess
from nlp.llm.client import OpenAICompatChatClient
from nlp.llm.config_resolver import resolve_server_config, resolve_request_config

# Standard utilities
from pathlib import Path
import atexit

def _resolve_path(p: str, project_root: Path) -> Path:
    """
    Resolve a path string to an absolute path
    - Expands ~
    - If relative, resolve it against the project roo.
    """
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (project_root / pp).resolve()

def build_container(cfg):
    """
    Dependency container builder
    Responsibility:
     - Takes a fully loaded config object
     - Constructs all shares services exactly one
     - Wires dependencies together
     - Returns a dictionary of ready-to-use services
    """

    # Determine the project root (used for resolving relative paths)
    project_root = Path(__file__).resolve().parents[1]

    # --------- Input layer ----------
    # DOCX loader for reading student submissions
    loader = DocxLoader(
        strip_whitespace=True, 
        keep_empty_paragraphs=False
    )

    # --------- GED (Grammer Error Detection) ----------
    # Load the GED BERT model
    ged_detector = GedBertDetector(model_name=cfg.ged.model_name)

    # Wrap the GED grammar detector in a service abstraction
    ged_service = GedService(detector=ged_detector)

    # ---------- LLM wiring (server mode) -----------
    server_proc = None
    if cfg.llama.llama_backend == "server":
        # Resolve llama-server binary path
        server_bin = _resolve_path(
            cfg.llama.llama_server_bin_path, 
            project_root
        )

        # Resolve GGUF model path
        model_path = Path(
            cfg.llama.llama_gguf_path
        ).expanduser().resolve()

        # Resolve optional multimodal projection file
        mmproj_path = None
        if cfg.llama.llama_mmproj_path:
            mmproj_path = Path(
                cfg.llama.llama_mmproj_path
            ).expanduser().resolve()

        server_cfg = resolve_server_config(
            cfg,
            server_overrides={
                "server_bin": server_bin,
                "model_path": model_path,
                "mmproj_path": mmproj_path,
            },
        )

        # Create the Llama server process wrapper
        server_proc = LlamaServerProcess(cfg=server_cfg)

        # Start the llama server
        server_proc.start()

        # Ensure the server is stopped cleanly on program exit
        atexit.register(server_proc.stop)

    # ---------- LLM client + service -----------

    # OpenAI-compativble HTTP client pointing at the local Llama server
    default_req = resolve_request_config("default", cfg)
    client = OpenAICompatChatClient(
        chat_url=cfg.llama.llama_server_url,
        model_name=cfg.llama.llama_model_alias,
        timeout_s=120,
        temperature=default_req.temperature,
    )

    # Higher level LLM service with model-family-specific logic
    llm_service = LlmService(
        client=client,
        model_family=cfg.llama.llama_model_family,
        app_cfg=cfg,
    )

    # ---------- Writing out assessments and explainability ----------

    # Recorder for explanations, decisions and reasoning traces
    explainability = ExplainabilityRecorder.new(
        run_cfg=cfg.run,
        ged_cfg=cfg.ged,
        llama_cfg=cfg.llama,
    )

    # Writer for persisting explainability output to disk
    explain_writer = ExplainabilityWriter(
        cfg.paths.explained_txt_folder
    )

    # DOCX output service (handles formatting and author metadata)
    docx_out = DocxOutputService(
        author=cfg.run.author
    )

    # ---- RETURN CONTAINER -----
    # Return all constructed services in a single lookup dictionary
    return {
        "loader": loader,
        "ged": ged_service,
        "cfg": cfg,
        "llm": llm_service,
        "llama-server": server_proc,
        "explain": explainability,
        "explain_writer": explain_writer,
        "docx_out": docx_out,
    }
