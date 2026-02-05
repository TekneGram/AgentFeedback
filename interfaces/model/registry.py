from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    backend: str
    model_family: str
    base_n_ctx: int
    min_ram_gb: int
    min_vram_gb: int
    notes: str

    hf_repo_id: str
    hf_filename: str
    mmproj_filename: Optional[str] = None
