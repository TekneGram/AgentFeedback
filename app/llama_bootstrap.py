from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from huggingface_hub import hf_hub_download
import os
import sys
from platformdirs import user_data_dir
from shutil import which, copy2
from helpers.llama_build import build_llama_server

# Determines where the app data should live
# In dev mode, uses .appdata
# In prod uses the OS-standard user data directory.
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

# Ensures the main GGUF model file exists locally and returns a valud path if already present.
# Otherwise, downloads if from Hugging Face
# Notmalizes the file location if HF creates nested paths.
# Returns the final local GGUF path.
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

# Ensures the multimodal projection file exists (if required)
# Return None if the model doesn't need one.
# Reuses an existing valid file if present.
# Otherwise, downloads it from Hugging Face
# Return the local mmproj path
def ensure_mmproj(cfg, models_dir: Path) -> Path | None:
    if not cfg.llama.hf_mmproj_filename:
        return None
    if not cfg.llama.hf_repo_id:
        raise ValueError("Need hf_repo_id + hf_mmproj_filename to download mmproj on first run.")

    models_dir.mkdir(parents=True, exist_ok=True)
    local_target = models_dir / cfg.llama.hf_mmproj_filename
    if local_target.exists() and local_target.stat().st_size > 0:
        return local_target

    downloaded = hf_hub_download(
        repo_id=cfg.llama.hf_repo_id,
        filename=cfg.llama.hf_mmproj_filename,
        revision=cfg.llama.hf_revision,
        local_dir=str(models_dir),
        token=True,
    )
    downloaded = Path(downloaded)
    if downloaded != local_target:
        downloaded.replace(local_target)
    return local_target

# Ensures the llama-server binary exists
# Uses an existing binary if found.
# Otherwise builds is locally via build_llama_server
# Return the path to teh server executable.
def ensure_llama_server_bin(app_cfg) -> Path:
    server = get_app_base_dir("EssayLens", "TekneGram") / "bin" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    if server.exists() and server.stat().st_size > 0:
        return server
    
    return build_llama_server(server, metal=False)

# Fully prepares the llama runtime
def bootstrap_llama(app_cfg):
    # Decide an app data dir (Electron later can pass its own)
    
    # Resolve the app base directory
    base = get_app_base_dir("EssayLens", "TekneGram")

    # Ensure the model files are available
    models_dir = base / "models"
    gguf_path = ensure_gguf(app_cfg, models_dir)
    mmproj_path = ensure_mmproj(app_cfg, models_dir)

    # Ensure the llama server binary exists
    server_bin = ensure_llama_server_bin(app_cfg)

    # Update the llama config with resolved file paths
    new_llama = replace(
        app_cfg.llama,
        llama_gguf_path=str(gguf_path),
        llama_server_bin_path=str(server_bin),
        llama_mmproj_path=str(mmproj_path) if mmproj_path else None,
    )

    # Validate the resolved configuration
    new_llama.validate_resolved()

    # Return a new app configuration with updated llama settings.
    return replace(app_cfg, llama=new_llama)
