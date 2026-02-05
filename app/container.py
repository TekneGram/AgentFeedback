from inout.docx_loader import DocxLoader
from nlp.ged_bert import GedBertDetector
from services.ged_service import GedService
from services.llm_service import LlmService
from services.explainability import ExplainabilityRecorder
from services.docx_output_service import DocxOutputService

from nlp.llm.server_process import LlamaServerProcess
from nlp.llm.client import OpenAICompatChatClient
from pathlib import Path
from urllib.parse import urlparse
import atexit
from inout.explainability_writer import ExplainabilityWriter

def _resolve_path(p: str, project_root: Path) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (project_root / pp).resolve()

def build_container(cfg):
    """
    Pattern-preserving container:
     - takes cfg
     - constructs shared dependencies once
     - returns dict of services
    """
    project_root = Path(__file__).resolve().parents[1]

    loader = DocxLoader(strip_whitespace=True, keep_empty_paragraphs=False)
    ged_detector = GedBertDetector(model_name=cfg.ged.model_name)
    ged_service = GedService(detector=ged_detector)

    # LLM wiring (server mode)
    def _server_from_config(llama_cfg):
        server_bin = _resolve_path(llama_cfg.llama_server_bin_path, project_root)
        model_path = Path(llama_cfg.llama_gguf_path).expanduser().resolve()
        mmproj_path = None
        if llama_cfg.llama_mmproj_path:
            mmproj_path = Path(llama_cfg.llama_mmproj_path).expanduser().resolve()
        parsed = urlparse(llama_cfg.llama_server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
        server_proc = LlamaServerProcess(
            server_bin=server_bin,
            model_path=model_path,
            model_alias=llama_cfg.llama_model_alias,
            mmproj_path=mmproj_path,
            host=host,
            port=port,
            n_ctx=llama_cfg.llama_n_ctx,
            n_threads=None,
        )
        server_proc.start()
        atexit.register(server_proc.stop)
        return server_proc

    server_proc_small = _server_from_config(cfg.llama_small)
    server_proc_big = _server_from_config(cfg.llama_big)

    small_client = OpenAICompatChatClient(
        chat_url=cfg.llama_small.llama_server_url,
        model_name=cfg.llama_small.llama_model_alias,
        timeout_s=120,
        temperature=0.0,
    )
    big_client = OpenAICompatChatClient(
        chat_url=cfg.llama_big.llama_server_url,
        model_name=cfg.llama_big.llama_model_alias,
        timeout_s=120,
        temperature=0.0,
    )
    llm_service = LlmService(
        small_client=small_client,
        big_client=big_client,
        small_model_family=cfg.llama_small.llama_model_family,
        big_model_family=cfg.llama_big.llama_model_family,
    )
    explainability = ExplainabilityRecorder.new(
        run_cfg=cfg.run,
        ged_cfg=cfg.ged,
        llama_small_cfg=cfg.llama_small,
        llama_big_cfg=cfg.llama_big,
    )
    explain_writer = ExplainabilityWriter(cfg.paths.explained_txt_folder)
    docx_out = DocxOutputService(author=cfg.run.author)

    return {
        "loader": loader,
        "ged": ged_service,
        "cfg": cfg,
        "llm": llm_service,
        "llama-server-small": server_proc_small,
        "llama-server-big": server_proc_big,
        "explain": explainability,
        "explain_writer": explain_writer,
        "docx_out": docx_out,
    }
