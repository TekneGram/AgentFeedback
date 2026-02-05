from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from config.ged_config import GedConfig
from config.paths_config import PathsConfig
from config.run_config import RunConfig
from config.llama_config import LlamaConfig

@dataclass(frozen=True, slots=True)
class AppConfig:
    paths: PathsConfig
    run: RunConfig
    ged: GedConfig
    llama: LlamaConfig


def build_settings() -> AppConfig:

    paths = PathsConfig.from_strings(
        input_docx_folder="Assessment/in",
        output_docx_folder="Assessment/checked",
        explained_txt_folder="Assessment/explained"
    )
    paths.validate()
    paths.ensure_output_dirs()

    run = RunConfig.from_strings(
        author="Daniel Parsons",
        single_paragraph_mode=True,
        max_llm_corrections=5,
        include_edited_text_section_policy=True
    )

    ged = GedConfig.from_strings(
        model_name="gotutiyan/token-ged-bert-large-cased-bin",
        batch_size=8,
    )

    llama = LlamaConfig.from_strings(
        llama_backend="server",
        llama_gguf_path="", # empty until bootstrap
        llama_server_url="http://127.0.0.1:8080/v1/chat/completions",
        llama_server_model = "llama",
        llama_model_key="default",
        llama_model_display_name="Default Model",
        llama_model_alias="Default Model",
        llama_model_family="instruct",
        llama_n_ctx=4096,
        llama_server_bin_path=".appdata/bin/llama-server",
        hf_repo_id="",
        hf_filename="",
        hf_mmproj_filename=None,
    )

    return AppConfig(paths=paths, run=run, ged=ged, llama=llama)
