from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING
import logging

from nlp.llm.task_config import TASK_DEFAULTS

if TYPE_CHECKING:
    from app.settings import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LlamaServerConfig:
    server_bin: Path
    model_path: Path
    model_alias: str
    host: str
    port: int
    n_ctx: int
    n_threads: Optional[int]
    n_gpu_layers: Optional[int]
    n_batch: Optional[int]
    seed: Optional[int]
    rope_freq_base: Optional[float]
    rope_freq_scale: Optional[float]
    mmproj_path: Optional[Path]


@dataclass(frozen=True, slots=True)
class LlmRequestConfig:
    max_tokens: int
    temperature: float
    top_p: Optional[float]
    top_k: Optional[int]
    repeat_penalty: Optional[float]
    seed: Optional[int]
    stop: Optional[list[str]]
    response_format: Optional[dict]
    stream: Optional[bool]


def _apply_overrides(values: dict[str, Any], overrides: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not overrides:
        return values
    unknown = sorted(set(overrides.keys()) - set(values.keys()))
    if unknown:
        raise ValueError(f"Unknown override keys: {', '.join(unknown)}")
    for key, val in overrides.items():
        values[key] = val
    return values


def resolve_server_config(
    app_cfg: "AppConfig",
    *,
    server_overrides: Optional[Mapping[str, Any]] = None,
) -> LlamaServerConfig:
    llama = app_cfg.llama
    values: dict[str, Any] = {
        "server_bin": Path(llama.llama_server_bin_path).expanduser().resolve()
        if llama.llama_server_bin_path
        else None,
        "model_path": Path(llama.llama_gguf_path).expanduser().resolve(),
        "model_alias": llama.llama_model_alias,
        "host": llama.llama_host,
        "port": llama.llama_port,
        "n_ctx": llama.llama_n_ctx,
        "n_threads": llama.llama_n_threads,
        "n_gpu_layers": llama.llama_n_gpu_layers,
        "n_batch": llama.llama_n_batch,
        "seed": llama.llama_seed,
        "rope_freq_base": llama.llama_rope_freq_base,
        "rope_freq_scale": llama.llama_rope_freq_scale,
        "mmproj_path": Path(llama.llama_mmproj_path).expanduser().resolve()
        if llama.llama_mmproj_path
        else None,
    }

    values = _apply_overrides(values, server_overrides)

    for key in ("server_bin", "model_path", "mmproj_path"):
        val = values.get(key)
        if val is not None and not isinstance(val, Path):
            values[key] = Path(val).expanduser().resolve()

    if values["server_bin"] is None:
        raise ValueError("llama_server_bin_path is required for server backend")

    cfg = LlamaServerConfig(**values)
    logger.info("Resolved LlamaServerConfig: %s", cfg)
    return cfg


def resolve_request_config(
    task_name: str,
    app_cfg: "AppConfig",
    *,
    request_overrides: Optional[Mapping[str, Any]] = None,
) -> LlmRequestConfig:
    llama = app_cfg.llama
    values: dict[str, Any] = {
        "max_tokens": llama.default_max_tokens,
        "temperature": llama.default_temperature,
        "top_p": llama.default_top_p,
        "top_k": llama.default_top_k,
        "repeat_penalty": llama.default_repeat_penalty,
        "seed": llama.default_seed,
        "stop": llama.default_stop,
        "response_format": llama.default_response_format,
        "stream": llama.default_stream,
    }

    task_defaults = TASK_DEFAULTS.get(task_name)
    if task_defaults:
        values = _apply_overrides(values, task_defaults)
    values = _apply_overrides(values, request_overrides)

    cfg = LlmRequestConfig(**values)
    logger.debug("Resolved LlmRequestConfig for %s: %s", task_name, cfg)
    return cfg
