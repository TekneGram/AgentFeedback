from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaModelSpec:
    key: str
    display_name: str
    hf_repo_id: str
    hf_filename: str
    mmproj_filename: str | None
    min_ram_gb: int
    min_vram_gb: int
    notes: str


MODEL_SPECS: list[LlamaModelSpec] = [
    LlamaModelSpec(
        key="qwen3_4b_instruct_q8",
        display_name="Qwen3 4B Q8_0 Instruct",
        hf_repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        hf_filename="Qwen3-4B-Instruct-2507-Q8_0.gguf",
        mmproj_filename=None,
        min_ram_gb=12,
        min_vram_gb=6,
        notes="CPU/GPU friendly; good quality for 4B.",
    ),
    LlamaModelSpec(
        key="qwen3_4b_thinking_q8",
        display_name="Qwen3 4B Q8_0 Thinking",
        hf_repo_id="unsloth/Qwen3-4B-Thinking-2507-GGUF",
        hf_filename="Qwen3-4B-Thinking-2507-Q8_0.gguf",
        mmproj_filename=None,
        min_ram_gb=12,
        min_vram_gb=6,
        notes="Thinking variant; slower but stronger reasoning.",
    ),
    LlamaModelSpec(
        key="qwen3_8b_vl_instruct_q8",
        display_name="Qwen3 8B Q8_0 Instruct (VL)",
        hf_repo_id="unsloth/Qwen3-VL-8B-Instruct-GGUF",
        hf_filename="Qwen3-VL-8B-Instruct-Q8_0.gguf",
        mmproj_filename="mmproj-F16.gguf",
        min_ram_gb=20,
        min_vram_gb=10,
        notes="VL model; needs mmproj for vision tasks.",
    ),
    LlamaModelSpec(
        key="qwen3_8b_vl_thinking_q8",
        display_name="Qwen3 8B Q8_0 Thinking (VL)",
        hf_repo_id="unsloth/Qwen3-VL-8B-Thinking-GGUF",
        hf_filename="Qwen3-VL-8B-Thinking-Q8_0.gguf",
        mmproj_filename="mmproj-F16.gguf",
        min_ram_gb=20,
        min_vram_gb=10,
        notes="VL thinking variant; highest quality if it fits.",
    ),
]
