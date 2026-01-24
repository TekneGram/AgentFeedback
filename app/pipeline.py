from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# I do not create DocxLoader objects
# They are injected into this pipeline so I only need to type check it.
if TYPE_CHECKING:
    from inout.docx_loader import DocxLoader

@dataclass
class FeedbackPipeline:
    loader: "DocxLoader"
    # editor: object
    # ged: object
    # llm: object
    # features: object
    # choose: object
    # selector: object
    # explain_writer: object

    def run_on_file(self, docx_path: Path, cfg) -> None:
        raw_paragraphs = self.loader.load_paragraphs(docx_path)