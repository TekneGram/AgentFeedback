from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# I do not create DocxLoader objects
# They are injected into this pipeline so I only need to type check it.
if TYPE_CHECKING:
    from inout.docx_loader import DocxLoader
    from services.ged_service import GedService

@dataclass
class FeedbackPipeline:
    loader: "DocxLoader"
    ged: "GedService"
    # editor: object
    # llm: object
    # features: object
    # choose: object
    # selector: object
    # explain_writer: object

    def run_on_file(self, docx_path: Path, cfg) -> None:
        raw_paragraphs = self.loader.load_paragraphs(docx_path)
        # Pre-process the paragraphs

        # Score sentences for ged
        ged_results = self.ged.score(raw_paragraphs, batch_size=cfg.ged.batch_size)
        print(ged_results)
