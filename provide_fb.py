from app.settings import build_settings
from app.container import build_container
from app.pipeline import FeedbackPipeline
from app.llama_bootstrap import bootstrap_llama

def main():
    # Build config (paths/run/ged/llama)
    app_cfg = build_settings()

    # Ensure gguf + llama-server exist, and return updated cfg (paths resolved)
    app_cfg = bootstrap_llama(app_cfg)

    # Build all services/objects via a container
    deps = build_container(app_cfg)

    # Construct the pipeline and inject the dependencies as kwargs (named arguments)
    pipeline = FeedbackPipeline(**deps)

    # Run on all input docs
    for docx_path in app_cfg.paths.list_input_docx():
        pipeline.run_on_file(docx_path, app_cfg)
    


if __name__ == "__main__":
    main()