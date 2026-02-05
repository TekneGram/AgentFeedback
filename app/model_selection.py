from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import os
import psutil
import torch

from config.llama_models import LlamaModelSpec, MODEL_SPECS
from app.llama_bootstrap import get_app_base_dir


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    total_ram_gb: float
    cpu_count: int
    cuda_vram_gb: float | None
    is_mps: bool

    @property
    def summary(self) -> str:
        vram = f"{self.cuda_vram_gb:.1f} GB VRAM" if self.cuda_vram_gb is not None else "No CUDA VRAM"
        mps = "MPS available" if self.is_mps else "MPS unavailable"
        return f"RAM: {self.total_ram_gb:.1f} GB | CPU: {self.cpu_count} | {vram} | {mps}"


def get_hardware_info() -> HardwareInfo:
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_count = os.cpu_count() or 1
    cuda_vram_gb = None
    if torch.cuda.is_available():
        try:
            mem = torch.cuda.get_device_properties(0).total_memory
            cuda_vram_gb = mem / (1024 ** 3)
        except Exception:
            cuda_vram_gb = None
    is_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    return HardwareInfo(
        total_ram_gb=total_ram_gb,
        cpu_count=cpu_count,
        cuda_vram_gb=cuda_vram_gb,
        is_mps=is_mps,
    )


def _fits_model(spec: LlamaModelSpec, hw: HardwareInfo) -> bool:
    if hw.total_ram_gb < spec.min_ram_gb:
        return False
    if hw.cuda_vram_gb is None:
        return True
    return hw.cuda_vram_gb >= spec.min_vram_gb


def recommend_model(specs: list[LlamaModelSpec], hw: HardwareInfo) -> LlamaModelSpec:
    # Best quality = largest model that fits
    ranked = sorted(specs, key=lambda s: (s.min_ram_gb, s.min_vram_gb), reverse=True)
    for spec in ranked:
        if _fits_model(spec, hw):
            return spec
    return ranked[-1]


def _persist_path(base_dir: Path) -> Path:
    return base_dir / "config" / "llama_model.json"

def get_models_dir(base_dir: Path) -> Path:
    return base_dir / "models"

def is_model_downloaded(spec: LlamaModelSpec, models_dir: Path) -> bool:
    if spec.backend == "server":
        gguf = models_dir / spec.hf_filename
        return gguf.exists() and gguf.stat().st_size > 0
    return False

def list_downloaded_specs(specs: list[LlamaModelSpec], models_dir: Path) -> list[LlamaModelSpec]:
    return [s for s in specs if is_model_downloaded(s, models_dir)]

def list_available_for_download(specs: list[LlamaModelSpec], models_dir: Path) -> list[LlamaModelSpec]:
    return [s for s in specs if not is_model_downloaded(s, models_dir)]


def load_persisted_model_key(base_dir: Path) -> str | None:
    path = _persist_path(base_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    key = data.get("llama_model_key")
    return key if isinstance(key, str) and key.strip() else None


def persist_model_key(base_dir: Path, key: str) -> None:
    path = _persist_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"llama_model_key": key}, indent=2), encoding="utf-8")


def _format_spec_line(idx: int, spec: LlamaModelSpec, recommended_key: str) -> str:
    marker = " (Recommended)" if spec.key == recommended_key else ""
    return (
        f"{idx}. {spec.display_name}{marker} | "
        f"min RAM {spec.min_ram_gb} GB, min VRAM {spec.min_vram_gb} GB"
    )


def prompt_initial_action(has_downloads: bool, has_installed: bool) -> str:
    print("\nModel setup")
    if has_downloads:
        label = "Select an installed model (Recommended)" if has_installed else "Select a model (Recommended)"
        print(f"1. {label}")
        print("2. Download a new model")
        prompt = "Selection [default: 1]: "
        while True:
            raw = input(prompt).strip()
            if not raw or raw == "1":
                return "select"
            if raw == "2":
                return "download"
            print("Invalid selection. Enter 1 or 2.")
    else:
        label = "Select an installed model" if has_installed else "Select a model"
        print(f"1. {label}")
        prompt = "Selection [default: 1]: "
        while True:
            raw = input(prompt).strip()
            if not raw or raw == "1":
                return "select"
            print("Invalid selection. Enter 1.")


def prompt_model_choice_from_list(
    specs: list[LlamaModelSpec],
    recommended: LlamaModelSpec,
    persisted_key: str | None,
    hw: HardwareInfo,
    label: str,
) -> LlamaModelSpec:
    print(f"\nModel selection - {label}")
    print(hw.summary)
    print("Choose a model (press Enter for default):")
    for i, spec in enumerate(specs, start=1):
        print(_format_spec_line(i, spec, recommended.key))

    default_spec = None
    if persisted_key:
        default_spec = next((s for s in specs if s.key == persisted_key), None)
    if default_spec is None:
        default_spec = next((s for s in specs if s.key == recommended.key), None)
    if default_spec is None:
        default_spec = specs[0]
    prompt = f"Selection [default: {default_spec.display_name}]: "

    while True:
        raw = input(prompt).strip()
        if not raw:
            return default_spec
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(specs):
                return specs[idx - 1]
        print("Invalid selection. Enter a number from the list or press Enter for default.")


def select_model_and_update_config(app_cfg):
    base_dir = get_app_base_dir("EssayLens", "TekneGram")
    backend = "server"
    models_dir = get_models_dir(base_dir)
    hw = get_hardware_info()
    filtered_specs = [s for s in MODEL_SPECS if s.backend == backend]
    recommended = recommend_model(filtered_specs, hw)
    persisted_key = load_persisted_model_key(base_dir)

    # If persisted choice fits, treat it as the recommended default.
    if persisted_key:
        persisted_spec = next((s for s in filtered_specs if s.key == persisted_key), None)
        if persisted_spec and _fits_model(persisted_spec, hw):
            recommended = persisted_spec
        else:
            persisted_key = None

    downloaded_specs = list_downloaded_specs(filtered_specs, models_dir)
    downloadable_specs = list_available_for_download(filtered_specs, models_dir)

    action = "select"
    action = prompt_initial_action(
        has_downloads=bool(downloadable_specs),
        has_installed=bool(downloaded_specs),
    )

    if action == "select":
        selection_list = downloaded_specs or filtered_specs
        chosen = prompt_model_choice_from_list(
            selection_list,
            recommended,
            persisted_key,
            hw,
            label="installed models",
        )
    else:
        selection_list = downloadable_specs or filtered_specs
        chosen = prompt_model_choice_from_list(
            selection_list,
            recommended,
            persisted_key,
            hw,
            label="available downloads",
        )
    persist_model_key(base_dir, chosen.key)

    new_llama = replace(
        app_cfg.llama,
        llama_backend=backend,
        hf_repo_id=chosen.hf_repo_id,
        hf_filename=chosen.hf_filename,
        hf_mmproj_filename=chosen.mmproj_filename,
        llama_model_key=chosen.key,
        llama_model_display_name=chosen.display_name,
        llama_model_alias=chosen.display_name,
        llama_model_family=chosen.model_family,
        llama_n_ctx=chosen.base_n_ctx * 2 if chosen.model_family == "thinking" else chosen.base_n_ctx,
    )
    new_llama.validate()
    return replace(app_cfg, llama=new_llama)
