from __future__ import annotations

from typing import Protocol
from pathlib import Path
from interfaces.config.app_config import AppConfigShape


class Pipeline(Protocol):
    def run_on_file(self, docx_path: Path, cfg: AppConfigShape) -> None:
        ...
