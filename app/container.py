from inout.docx_loader import DocxLoader
from nlp.ged_bert import GedBertDetector
from services.ged_service import GedService
from services.llm_service import LlmService

def build_container(cfg):
    loader = DocxLoader(strip_whitespace=True, keep_empty_paragraphs=False)
    ged_detector = GedBertDetector(model_name=cfg.ged.model_name)
    ged_service = GedService(detector=ged_detector)

    return {
        "loader": loader,
        "ged": ged_service
    }