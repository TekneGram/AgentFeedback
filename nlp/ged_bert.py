from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

@dataclass
class GedSentenceResult:
    sentence: str
    has_error: bool

class GedBertDetector:
    """
    Fast sentence-level GED
    - Marks a sentence as error if ANY non-special token is predicted as ERROR
    - Avoids spaCy alignment
    """

    def __init__(
            self,
            model_name: str,
            device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.ERROR_ID = 1