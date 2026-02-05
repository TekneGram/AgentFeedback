from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaModelSpec:
    key: str
    display_name: str
    hf_repo_id: str
    hf_filename: str
    mmproj_filename: str | None
    backend: str
    model_family: str
    base_n_ctx: int
    min_ram_gb: int
    min_vram_gb: int
    param_size_b: int
    notes: str


MODEL_SPECS: list[LlamaModelSpec] = [
    LlamaModelSpec(
        key="qwen3_4b_instruct_q8",
        display_name="Qwen3 4B Q8_0 Instruct",
        hf_repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        hf_filename="Qwen3-4B-Instruct-2507-Q8_0.gguf",
        mmproj_filename=None,
        backend="server",
        model_family="instruct",
        base_n_ctx=4096,
        min_ram_gb=12,
        min_vram_gb=6,
        param_size_b=4,
        notes="CPU/GPU friendly; good quality for 4B.",
    ),
    LlamaModelSpec(
        key="qwen3_4b_thinking_q8",
        display_name="Qwen3 4B Q8_0 Thinking",
        hf_repo_id="unsloth/Qwen3-4B-Thinking-2507-GGUF",
        hf_filename="Qwen3-4B-Thinking-2507-Q8_0.gguf",
        mmproj_filename=None,
        backend="server",
        model_family="thinking",
        base_n_ctx=4096,
        min_ram_gb=12,
        min_vram_gb=6,
        param_size_b=4,
        notes="Thinking variant; slower but stronger reasoning.",
    ),
    LlamaModelSpec(
        key="qwen3_8b_vl_instruct_q8",
        display_name="Qwen3 8B Q8_0 Instruct (VL)",
        hf_repo_id="unsloth/Qwen3-VL-8B-Instruct-GGUF",
        hf_filename="Qwen3-VL-8B-Instruct-Q8_0.gguf",
        mmproj_filename="mmproj-F16.gguf",
        backend="server",
        model_family="instruct",
        base_n_ctx=4096,
        min_ram_gb=20,
        min_vram_gb=10,
        param_size_b=8,
        notes="VL model; needs mmproj for vision tasks.",
    ),
    LlamaModelSpec(
        key="qwen3_vl_30B_A3B_instruct",
        display_name="Qwen3 30B A3B Q4_K_M Instruct (VL)",
        hf_repo_id="Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",
        hf_filename="Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
        mmproj_filename="mmproj-F16.gguf",
        backend="server",
        model_family="instruct",
        base_n_ctx=4096,
        min_ram_gb=22,
        min_vram_gb=12,
        param_size_b=30,
        notes="VL model; needs mmproj for vision tasks.",
    ),
    LlamaModelSpec(
        key="qwen3_8b_vl_thinking_q8",
        display_name="Qwen3 8B Q8_0 Thinking (VL)",
        hf_repo_id="unsloth/Qwen3-VL-8B-Thinking-GGUF",
        hf_filename="Qwen3-VL-8B-Thinking-Q8_0.gguf",
        mmproj_filename="mmproj-F16.gguf",
        backend="server",
        model_family="thinking",
        base_n_ctx=4096,
        min_ram_gb=20,
        min_vram_gb=10,
        param_size_b=8,
        notes="VL thinking variant; highest quality if it fits.",
    ),
    LlamaModelSpec(
        key="gemma-3-1b-it",
        display_name="Gemma3 1B IT",
        hf_repo_id="bartowski/google_gemma-3-1b-it-GGUF",
        hf_filename="google_gemma-3-1b-it-bf16.gguf",
        mmproj_filename=None,
        backend="server",
        model_family="instruct",
        base_n_ctx=4096,
        min_ram_gb=6,
        min_vram_gb=4,
        param_size_b=1,
        notes="CPU/GPU friendly; good quality for 1B.",
    ),
]
