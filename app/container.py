from inout.docx_loader import DocxLoader
from nlp.ged_bert import GedBertDetector

def build_container(cfg):
    loader = DocxLoader(strip_whitespace=True, keep_empty_paragraphs=False)
    ged_bert = GedBertDetector(model_name=cfg.ged.model_name)

    return {
        "loader": loader,
        "ged_bert": ged_bert
    }