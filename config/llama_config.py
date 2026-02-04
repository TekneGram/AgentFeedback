from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class LlamaConfig:
    llama_backend: str          # "local" or "server"
    llama_gguf_path: str        # resolved after bootstrap
    llama_server_url: str
    llama_server_model: str
    llama_model_key: str
    llama_model_display_name: str

    hf_repo_id: str | None = None
    hf_filename: str | None = None
    hf_revision: str | None = None
    hf_mmproj_filename: str | None = None

    llama_server_bin_path: str | None = None # resolved/bundled path to llama-server
    llama_mmproj_path: str | None = None

    def validate_resolved(self) -> None:
        # Call this AFTER bootstrapping
        if not self.llama_gguf_path or not Path(self.llama_gguf_path).exists():
            raise ValueError("Resolved gguf path missing or does not exist")
        if self.llama_mmproj_path:
            mmproj = Path(self.llama_mmproj_path)
            if not mmproj.exists():
                raise ValueError("Resolved mmproj path missing or does not exist")

    def validate(self) -> None:
        if not isinstance(self.llama_backend, str) or not self.llama_backend.strip():
            raise ValueError("LlamaConfig.llama_backend must be a non-empty string.")
        if not isinstance(self.llama_server_model, str) or not self.llama_server_model.strip():
            raise ValueError("LLamaConfig.llama_server_model must be a non-empty string.")
        if not isinstance(self.llama_server_bin_path, str) or not self.llama_server_bin_path.strip():
            raise ValueError("LlamaConfig.llama_server_bin_path must be a non-empty string.")
        if not isinstance(self.llama_model_key, str) or not self.llama_model_key.strip():
            raise ValueError("LlamaConfig.llama_model_key must be a non-empty string.")
        if not isinstance(self.llama_model_display_name, str) or not self.llama_model_display_name.strip():
            raise ValueError("LlamaConfig.llama_model_display_name must be a non-empty string.")
    
    @staticmethod
    def from_strings(
            llama_backend: str,
            llama_gguf_path: str,
            llama_server_url: str,
            llama_server_model: str,
            llama_model_key: str,
            llama_model_display_name: str,
            llama_server_bin_path: str | None,
            hf_repo_id: str | None,
            hf_filename: str | None,
            hf_mmproj_filename: str | None,
    ) -> "LlamaConfig":
        cfg = LlamaConfig(
            llama_backend=llama_backend, 
            llama_gguf_path=llama_gguf_path,
            llama_server_url=llama_server_url,
            llama_server_model=llama_server_model,
            llama_model_key=llama_model_key,
            llama_model_display_name=llama_model_display_name,
            llama_server_bin_path=llama_server_bin_path,
            hf_repo_id=hf_repo_id,
            hf_filename=hf_filename,
            hf_mmproj_filename=hf_mmproj_filename,
            )
        cfg.validate()
        return cfg
