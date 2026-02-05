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


def load_persisted_model_keys(base_dir: Path) -> tuple[str | None, str | None]:
    path = _persist_path(base_dir)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    small_key = data.get("llama_small_model_key")
    big_key = data.get("llama_big_model_key")
    small_key = small_key if isinstance(small_key, str) and small_key.strip() else None
    big_key = big_key if isinstance(big_key, str) and big_key.strip() else None
    return small_key, big_key


def persist_model_keys(base_dir: Path, *, small_key: str, big_key: str) -> None:
    path = _persist_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "llama_small_model_key": small_key,
                "llama_big_model_key": big_key,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _format_spec_line(idx: int, spec: LlamaModelSpec, recommended_key: str) -> str:
    marker = " (Recommended)" if spec.key == recommended_key else ""
    return (
        f"{idx}. {spec.display_name}{marker} | "
        f"{spec.param_size_b}B | min RAM {spec.min_ram_gb} GB, min VRAM {spec.min_vram_gb} GB"
    )


def prompt_initial_action(tier_label: str, has_downloads: bool, has_installed: bool) -> str:
    print(f"\nModel setup - {tier_label}")
    if has_downloads:
        if has_installed:
            print(f"1. Select an installed {tier_label} model (Recommended)")
            print(f"2. Download a new {tier_label} model")
            prompt = "Selection [default: 1]: "
            default = "1"
        else:
            print(f"1. Download a new {tier_label} model (Recommended)")
            print(f"2. Select a {tier_label} model")
            prompt = "Selection [default: 1]: "
            default = "1"
        while True:
            raw = input(prompt).strip()
            if not raw:
                raw = default
            if raw == "1":
                return "select" if has_installed else "download"
            if raw == "2":
                return "download" if has_installed else "select"
            print("Invalid selection. Enter 1 or 2.")
    else:
        label = f"Select an installed {tier_label} model" if has_installed else f"Select a {tier_label} model"
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


def _filter_specs_by_size(specs: list[LlamaModelSpec], *, min_b: int | None, max_b: int | None) -> list[LlamaModelSpec]:
    out: list[LlamaModelSpec] = []
    for spec in specs:
        if min_b is not None and spec.param_size_b < min_b:
            continue
        if max_b is not None and spec.param_size_b > max_b:
            continue
        out.append(spec)
    return out


def _select_for_tier(
    *,
    tier_label: str,
    specs: list[LlamaModelSpec],
    models_dir: Path,
    hw: HardwareInfo,
    persisted_key: str | None,
) -> LlamaModelSpec:
    if not specs:
        raise ValueError(f"No models available for tier: {tier_label}")
    recommended = recommend_model(specs, hw)

    # If persisted choice fits, treat it as the recommended default.
    if persisted_key:
        persisted_spec = next((s for s in specs if s.key == persisted_key), None)
        if persisted_spec and _fits_model(persisted_spec, hw):
            recommended = persisted_spec
        else:
            persisted_key = None

    downloaded_specs = list_downloaded_specs(specs, models_dir)
    downloadable_specs = list_available_for_download(specs, models_dir)

    action = prompt_initial_action(
        tier_label,
        has_downloads=bool(downloadable_specs),
        has_installed=bool(downloaded_specs),
    )

    if action == "select":
        selection_list = downloaded_specs or specs
        label = f"{tier_label} installed models"
    else:
        selection_list = downloadable_specs or specs
        label = f"{tier_label} available downloads"

    return prompt_model_choice_from_list(
        selection_list,
        recommended,
        persisted_key,
        hw,
        label=label,
    )


def select_model_and_update_config(app_cfg):
    base_dir = get_app_base_dir("EssayLens", "TekneGram")
    backend = "server"
    models_dir = get_models_dir(base_dir)
    hw = get_hardware_info()
    all_specs = [s for s in MODEL_SPECS if s.backend == backend]
    small_specs = _filter_specs_by_size(all_specs, min_b=None, max_b=4)
    big_specs = _filter_specs_by_size(all_specs, min_b=4, max_b=None)

    persisted_small_key, persisted_big_key = load_persisted_model_keys(base_dir)

    small_choice = _select_for_tier(
        tier_label="small",
        specs=small_specs,
        models_dir=models_dir,
        hw=hw,
        persisted_key=persisted_small_key,
    )
    big_choice = _select_for_tier(
        tier_label="big",
        specs=big_specs,
        models_dir=models_dir,
        hw=hw,
        persisted_key=persisted_big_key,
    )

    persist_model_keys(base_dir, small_key=small_choice.key, big_key=big_choice.key)

    new_llama_small = replace(
        app_cfg.llama_small,
        llama_backend=backend,
        hf_repo_id=small_choice.hf_repo_id,
        hf_filename=small_choice.hf_filename,
        hf_mmproj_filename=small_choice.mmproj_filename,
        llama_model_key=small_choice.key,
        llama_model_display_name=small_choice.display_name,
        llama_model_alias=small_choice.display_name,
        llama_model_family=small_choice.model_family,
        llama_n_ctx=small_choice.base_n_ctx * 2 if small_choice.model_family == "thinking" else small_choice.base_n_ctx,
    )
    new_llama_small.validate()

    new_llama_big = replace(
        app_cfg.llama_big,
        llama_backend=backend,
        hf_repo_id=big_choice.hf_repo_id,
        hf_filename=big_choice.hf_filename,
        hf_mmproj_filename=big_choice.mmproj_filename,
        llama_model_key=big_choice.key,
        llama_model_display_name=big_choice.display_name,
        llama_model_alias=big_choice.display_name,
        llama_model_family=big_choice.model_family,
        llama_n_ctx=big_choice.base_n_ctx * 2 if big_choice.model_family == "thinking" else big_choice.base_n_ctx,
    )
    new_llama_big.validate()

    return replace(app_cfg, llama_small=new_llama_small, llama_big=new_llama_big)
