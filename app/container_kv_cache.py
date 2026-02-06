from inout.docx_loader import DocxLoader
from nlp.ged_bert import GedBertDetector
from services.ged_service import GedService
from services_kv_cache.llm_service_kv import LlmServiceKV
from services.explainability import ExplainabilityRecorder
from services.docx_output_service import DocxOutputService

from nlp_kv_cache.llm.kv_client import KvCacheClient
from pathlib import Path
from inout.explainability_writer import ExplainabilityWriter


def _resolve_path(p: str, project_root: Path) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (project_root / pp).resolve()


def build_container_kv_cache(cfg):
    """
    KV-cache container:
     - takes cfg
     - constructs shared dependencies once
     - returns dict of services
    """
    project_root = Path(__file__).resolve().parents[1]

    loader = DocxLoader(strip_whitespace=True, keep_empty_paragraphs=False)
    ged_detector = GedBertDetector(model_name=cfg.ged.model_name)
    ged_service = GedService(detector=ged_detector)

    model_path = Path(cfg.llama.llama_gguf_path).expanduser().resolve()
    kv_client = KvCacheClient(
        model_path=str(model_path),
        n_ctx=cfg.llama.llama_n_ctx,
        n_threads=None,
        n_gpu_layers=None,
        thinking_mode="no_think",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
    )
    llm_service = LlmServiceKV(kv_client=kv_client, model_family=cfg.llama.llama_model_family)
    explainability = ExplainabilityRecorder.new(
        run_cfg=cfg.run,
        ged_cfg=cfg.ged,
        llama_cfg=cfg.llama,
    )
    explain_writer = ExplainabilityWriter(cfg.paths.explained_txt_folder)
    docx_out = DocxOutputService(author=cfg.run.author)

    return {
        "loader": loader,
        "ged": ged_service,
        "cfg": cfg,
        "llm": llm_service,
        "explain": explainability,
        "explain_writer": explain_writer,
        "docx_out": docx_out,
    }
