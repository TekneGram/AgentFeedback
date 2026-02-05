from __future__ import annotations

from typing import Protocol
from pathlib import Path


class DocxOutput(Protocol):
    def build_report_with_header_and_body(
        self,
        *,
        output_path: Path,
        original_paragraphs: list[str],
        edited_text: str,
        header_lines: list[str],
        edited_body_text: str,
        corrected_body_text: str,
        feedback_paragraphs: list[str],
        include_edited_text: bool,
    ) -> None:
        ...
