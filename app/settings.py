from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from config.ged_config import GedConfig
from config.paths_config import PathsConfig
from config.run_config import RunConfig

@dataclass(frozen=True, slots=True)
class AppConfig:
    paths: PathsConfig
    run: RunConfig
    ged: GedConfig


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
        model_name="gotutiyan/token-get-bert-large-cased-bin",
        batch_size=8,
    )

    return AppConfig(paths=paths, run=run, ged=ged)