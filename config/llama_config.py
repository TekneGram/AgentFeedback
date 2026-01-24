from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LlamaConfig:
    llama_backend: str
    llama_gguf_path: str
    llama_server_url: str
    llama_server_model: str

    def validate(self) -> None:
        if not isinstance(self.llama_backend, str) or not self.llama_backend.strip():
            raise ValueError("LlamaConfig.llama_backend must be a non-empty string.")
        if not isinstance(self.llama_gguf_path, str) or not self.llama_gguf_path.strip():
            raise ValueError("LLamaConfig.llama_gguf_path must be a non-empty string.")
        if not isinstance(self.llama_server_url, str) or not self.llama_server_url.strip():
            raise ValueError("LLamaConfig.llama_server_url must be a non-empty string.")
        if not isinstance(self.llama_server_model, str) or not self.llama_server_model.strip():
            raise ValueError("LLamaConfig.llama_server_model must be a non-empty string.")
    
    @staticmethod
    def from_strings(
            llama_backend: str,
            llama_gguf_path: str,
            llama_server_url: str,
            llama_server_model: str
    ) -> "LlamaConfig":
        cfg = LlamaConfig(
            llama_backend=llama_backend, 
            llama_gguf_path=llama_gguf_path,
            llama_server_url=llama_server_url,
            llama_server_model=llama_server_model
            )
        cfg.validate()
        return cfg