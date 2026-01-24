from inout.docx_loader import DocxLoader

def build_container(cfg):
    loader = DocxLoader(strip_whitespace=True, keep_empty_paragraphs=False)

    return {
        "loader": loader,
    }