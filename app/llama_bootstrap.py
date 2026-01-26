from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import sys
from platformdirs import user_data_dir
from shutil import which, copy2
from helpers.llama_build import build_llama_server

def _get_app_base_dir(app_name: str, org: str) -> Path:
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

def ensure_gguf(cfg, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)

    # If already resolved and present, keep it
    if cfg.llama.llama_gguf_path:
        p = Path(cfg.llama.llama_gguf_path)
        if p.exists() and p.stat().st_size > 0:
            return p
        
    if not cfg.llama.hf_repo_id or not cfg.llama.hf_filename:
        raise ValueError("Need hf_repo_id + hf_filename to download gguf on first run.")
    
    local_target = models_dir / cfg.llama.hf_filename
    if local_target.exists() and local_target.stat().st_size > 0:
        return local_target
    
    downloaded = hf_hub_download(
        repo_id=cfg.llama.hf_repo_id,
        filename=cfg.llama.hf_filename,
        revision=cfg.llama.hf_revision,
        local_dir=str(models_dir),
        token=True,
    )
    downloaded = Path(downloaded)

    # Normalize location if HF created nested paths
    if downloaded != local_target:
        downloaded.replace(local_target)

    return local_target

def ensure_llama_server_bin(app_cfg) -> Path:
    server = _get_app_base_dir("EssayLens", "TekneGram") / "bin" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    if server.exists() and server.stat().st_size > 0:
        return server
    
    return build_llama_server(server, metal=False)


def bootstrap_llama(app_cfg):
    # Decide an app data dir (Electron later can pass its own)
    base = _get_app_base_dir("EssayLens", "TekneGram")
    models_dir = base / "models"

    gguf_path = ensure_gguf(app_cfg, models_dir)
    server_bin = ensure_llama_server_bin(app_cfg)

    new_llama = replace(
        app_cfg.llama,
        llama_gguf_path=str(gguf_path),
        llama_server_bin_path=str(server_bin)
    )

    new_llama.validate_resolved()
    return replace(app_cfg, llama=new_llama)
    #return app_cfg
