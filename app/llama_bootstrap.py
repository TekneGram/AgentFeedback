from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import sys
from platformdirs import user_data_dir
from shutil import which, copy2
from helpers.llama_build import build_llama_server
from config.llama_config import LlamaConfig

def get_app_base_dir(app_name: str, org: str) -> Path:
    # Explicit override for dev + Electron later
    override = os.getenv("APP_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    
    # Dev mode -> store inside the repo
    if os.getenv("DEV_MODE", "").strip() in {"1", "true", "True", "yes", "YES"}:
        project_root = Path(__file__).resolve().parents[1]
        return (project_root / ".appdata").resolve()
    
    # Prod mode -> OS-standard user data dir
    return Path(user_data_dir(app_name, org)).resolve()

def ensure_gguf(cfg: LlamaConfig, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)

    # If already resolved and present, keep it
    if cfg.llama_gguf_path:
        p = Path(cfg.llama_gguf_path)
        if p.exists() and p.stat().st_size > 0:
            return p
        
    if not cfg.hf_repo_id or not cfg.hf_filename:
        raise ValueError("Need hf_repo_id + hf_filename to download gguf on first run.")
    
    local_target = models_dir / cfg.hf_filename
    if local_target.exists() and local_target.stat().st_size > 0:
        return local_target
    
    downloaded = hf_hub_download(
        repo_id=cfg.hf_repo_id,
        filename=cfg.hf_filename,
        revision=cfg.hf_revision,
        local_dir=str(models_dir),
        token=True,
    )
    downloaded = Path(downloaded)

    # Normalize location if HF created nested paths
    if downloaded != local_target:
        downloaded.replace(local_target)

    return local_target

def ensure_mmproj(cfg: LlamaConfig, models_dir: Path) -> Path | None:
    if not cfg.hf_mmproj_filename:
        return None
    if not cfg.hf_repo_id:
        raise ValueError("Need hf_repo_id + hf_mmproj_filename to download mmproj on first run.")

    models_dir.mkdir(parents=True, exist_ok=True)
    local_target = models_dir / cfg.hf_mmproj_filename
    if local_target.exists() and local_target.stat().st_size > 0:
        return local_target

    downloaded = hf_hub_download(
        repo_id=cfg.hf_repo_id,
        filename=cfg.hf_mmproj_filename,
        revision=cfg.hf_revision,
        local_dir=str(models_dir),
        token=True,
    )
    downloaded = Path(downloaded)
    if downloaded != local_target:
        downloaded.replace(local_target)
    return local_target

def ensure_llama_server_bin(app_cfg) -> Path:
    server = get_app_base_dir("EssayLens", "TekneGram") / "bin" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    if server.exists() and server.stat().st_size > 0:
        return server
    
    return build_llama_server(server, metal=False)


def _bootstrap_single_llama(app_cfg, cfg: LlamaConfig, models_dir: Path) -> LlamaConfig:
    gguf_path = ensure_gguf(cfg, models_dir)
    mmproj_path = ensure_mmproj(cfg, models_dir)
    server_bin = ensure_llama_server_bin(app_cfg)

    new_llama = replace(
        cfg,
        llama_gguf_path=str(gguf_path),
        llama_server_bin_path=str(server_bin),
        llama_mmproj_path=str(mmproj_path) if mmproj_path else None,
    )

    new_llama.validate_resolved()
    return new_llama


def bootstrap_llama(app_cfg):
    # Decide an app data dir (Electron later can pass its own)
    base = get_app_base_dir("EssayLens", "TekneGram")
    models_dir = base / "models"
    new_llama_small = _bootstrap_single_llama(app_cfg, app_cfg.llama_small, models_dir)
    new_llama_big = _bootstrap_single_llama(app_cfg, app_cfg.llama_big, models_dir)

    return replace(app_cfg, llama_small=new_llama_small, llama_big=new_llama_big)
