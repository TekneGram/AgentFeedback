from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class LlamaConfig:
    llama_backend: str          # "server"
    llama_gguf_path: str        # resolved after bootstrap
    llama_server_url: str
    llama_server_model: str
    llama_model_key: str
    llama_model_display_name: str
    llama_model_alias: str
    llama_model_family: str
    llama_n_ctx: int
    llama_host: str
    llama_port: int
    llama_n_threads: int | None
    llama_n_gpu_layers: int | None
    llama_n_batch: int | None
    llama_seed: int | None
    llama_rope_freq_base: float | None
    llama_rope_freq_scale: float | None

    default_max_tokens: int
    default_temperature: float
    default_top_p: float | None
    default_top_k: int | None
    default_repeat_penalty: float | None
    default_seed: int | None
    default_stop: list[str] | None
    default_response_format: dict | None
    default_stream: bool | None

    hf_repo_id: str | None = None
    hf_filename: str | None = None
    hf_revision: str | None = None
    hf_mmproj_filename: str | None = None

    llama_server_bin_path: str | None = None # resolved/bundled path to llama-server
    llama_mmproj_path: str | None = None

    def validate_resolved(self) -> None:
        # Call this AFTER bootstrapping
        if self.llama_backend == "server":
            if not self.llama_gguf_path or not Path(self.llama_gguf_path).exists():
                raise ValueError("Resolved gguf path missing or does not exist")
            if self.llama_mmproj_path:
                mmproj = Path(self.llama_mmproj_path)
                if not mmproj.exists():
                    raise ValueError("Resolved mmproj path missing or does not exist")

    def validate(self) -> None:
        if not isinstance(self.llama_backend, str) or not self.llama_backend.strip():
            raise ValueError("LlamaConfig.llama_backend must be a non-empty string.")
        if self.llama_backend not in {"server"}:
            raise ValueError("LlamaConfig.llama_backend must be 'server'.")
        if not isinstance(self.llama_server_model, str) or not self.llama_server_model.strip():
            raise ValueError("LLamaConfig.llama_server_model must be a non-empty string.")
        if self.llama_backend == "server":
            if not isinstance(self.llama_server_bin_path, str) or not self.llama_server_bin_path.strip():
                raise ValueError("LlamaConfig.llama_server_bin_path must be a non-empty string.")
        if not isinstance(self.llama_model_key, str) or not self.llama_model_key.strip():
            raise ValueError("LlamaConfig.llama_model_key must be a non-empty string.")
        if not isinstance(self.llama_model_display_name, str) or not self.llama_model_display_name.strip():
            raise ValueError("LlamaConfig.llama_model_display_name must be a non-empty string.")
        if not isinstance(self.llama_model_alias, str) or not self.llama_model_alias.strip():
            raise ValueError("LlamaConfig.llama_model_alias must be a non-empty string.")
        if not isinstance(self.llama_model_family, str) or self.llama_model_family not in {"instruct", "thinking"}:
            raise ValueError("LlamaConfig.llama_model_family must be 'instruct' or 'thinking'.")
        if not isinstance(self.llama_n_ctx, int) or self.llama_n_ctx <= 0:
            raise ValueError("LlamaConfig.llama_n_ctx must be a positive integer.")
        if not isinstance(self.llama_host, str) or not self.llama_host.strip():
            raise ValueError("LlamaConfig.llama_host must be a non-empty string.")
        if not isinstance(self.llama_port, int) or self.llama_port <= 0:
            raise ValueError("LlamaConfig.llama_port must be a positive integer.")
        if self.llama_n_threads is not None and (not isinstance(self.llama_n_threads, int) or self.llama_n_threads <= 0):
            raise ValueError("LlamaConfig.llama_n_threads must be a positive integer or None.")
        if self.llama_n_gpu_layers is not None and not isinstance(self.llama_n_gpu_layers, int):
            raise ValueError("LlamaConfig.llama_n_gpu_layers must be an integer or None.")
        if self.llama_n_batch is not None and (not isinstance(self.llama_n_batch, int) or self.llama_n_batch <= 0):
            raise ValueError("LlamaConfig.llama_n_batch must be a positive integer or None.")
        if self.llama_seed is not None and not isinstance(self.llama_seed, int):
            raise ValueError("LlamaConfig.llama_seed must be an integer or None.")
        if self.llama_rope_freq_base is not None and not isinstance(self.llama_rope_freq_base, (int, float)):
            raise ValueError("LlamaConfig.llama_rope_freq_base must be a float or None.")
        if self.llama_rope_freq_scale is not None and not isinstance(self.llama_rope_freq_scale, (int, float)):
            raise ValueError("LlamaConfig.llama_rope_freq_scale must be a float or None.")
        if not isinstance(self.default_max_tokens, int) or self.default_max_tokens <= 0:
            raise ValueError("LlamaConfig.default_max_tokens must be a positive integer.")
        if not isinstance(self.default_temperature, (int, float)):
            raise ValueError("LlamaConfig.default_temperature must be a float.")
        if self.default_top_p is not None and not isinstance(self.default_top_p, float):
            raise ValueError("LlamaConfig.default_top_p must be a float or None.")
        if self.default_top_k is not None and not isinstance(self.default_top_k, int):
            raise ValueError("LlamaConfig.default_top_k must be an integer or None.")
        if self.default_repeat_penalty is not None and not isinstance(self.default_repeat_penalty, float):
            raise ValueError("LlamaConfig.default_repeat_penalty must be a float or None.")
        if self.default_seed is not None and not isinstance(self.default_seed, int):
            raise ValueError("LlamaConfig.default_seed must be an integer or None.")
        if self.default_stop is not None and not isinstance(self.default_stop, list):
            raise ValueError("LlamaConfig.default_stop must be a list[str] or None.")
        if self.default_response_format is not None and not isinstance(self.default_response_format, dict):
            raise ValueError("LlamaConfig.default_response_format must be a dict or None.")
        if self.default_stream is not None and not isinstance(self.default_stream, bool):
            raise ValueError("LlamaConfig.default_stream must be a bool or None.")
    
    @staticmethod
    def from_strings(
            llama_backend: str,
            llama_gguf_path: str,
            llama_server_url: str,
            llama_server_model: str,
            llama_model_key: str,
            llama_model_display_name: str,
            llama_model_alias: str,
            llama_model_family: str,
            llama_n_ctx: int,
            llama_host: str,
            llama_port: int,
            llama_n_threads: int | None,
            llama_n_gpu_layers: int | None,
            llama_n_batch: int | None,
            llama_seed: int | None,
            llama_rope_freq_base: float | None,
            llama_rope_freq_scale: float | None,
            default_max_tokens: int,
            default_temperature: float,
            default_top_p: float | None,
            default_top_k: int | None,
            default_repeat_penalty: float | None,
            default_seed: int | None,
            default_stop: list[str] | None,
            default_response_format: dict | None,
            default_stream: bool | None,
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
            llama_model_alias=llama_model_alias,
            llama_model_family=llama_model_family,
            llama_n_ctx=llama_n_ctx,
            llama_host=llama_host,
            llama_port=llama_port,
            llama_n_threads=llama_n_threads,
            llama_n_gpu_layers=llama_n_gpu_layers,
            llama_n_batch=llama_n_batch,
            llama_seed=llama_seed,
            llama_rope_freq_base=llama_rope_freq_base,
            llama_rope_freq_scale=llama_rope_freq_scale,
            default_max_tokens=default_max_tokens,
            default_temperature=default_temperature,
            default_top_p=default_top_p,
            default_top_k=default_top_k,
            default_repeat_penalty=default_repeat_penalty,
            default_seed=default_seed,
            default_stop=default_stop,
            default_response_format=default_response_format,
            default_stream=default_stream,
            llama_server_bin_path=llama_server_bin_path,
            hf_repo_id=hf_repo_id,
            hf_filename=hf_filename,
            hf_mmproj_filename=hf_mmproj_filename,
            )
        cfg.validate()
        return cfg
