from __future__ import annotations

from dataclasses import dataclass
from config.paths_config import PathsConfig
from config.run_config import RunConfig
from config.ged_config import GedConfig
from config.llama_config import LlamaConfig


@dataclass(frozen=True)
class AppConfigShape:
    paths: PathsConfig
    run: RunConfig
    ged: GedConfig
    llama: LlamaConfig
