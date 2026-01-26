from app.settings import build_settings
from app.container import build_container
from app.pipeline import FeedbackPipeline
from app.llama_bootstrap import bootstrap_llama

from utils.terminal_ui import Color, type_print

def main():
    # Build config (paths/run/ged/llama)
    type_print("Building the app settings", color=Color.BLUE)
    app_cfg = build_settings()

    # Ensure gguf + llama-server exist, and return updated cfg (paths resolved)
    type_print("Bootstrapping Llama", color=Color.BLUE)
    app_cfg = bootstrap_llama(app_cfg)
    print(f"This is the config: {app_cfg}")

    # Build all services/objects via a container
    type_print("Building all the services and loading the LLM", color=Color.BLUE)
    deps = build_container(app_cfg)

    # Construct the pipeline and inject the dependencies as kwargs (named arguments)
    # pipeline = FeedbackPipeline(**deps)
    type_print("Constructing the nlp pipeline.", color=Color.BLUE)
    pipeline = FeedbackPipeline(
        loader=deps["loader"],
        ged=deps["ged"],
        llm=deps["llm"],
        explain=deps["explain"],
        explain_writer=deps["explain_writer"],
        docx_out=deps["docx_out"],
    )

    # # TESTS
    # # Basic chat:
    # reply = pipeline.llm.answer("Tell me something interesting")
    # print(reply)

    # # Streaming chat
    # pipeline.llm.stream_answer("Tell me a joke that is not about Pavlov's dog or librarians!")
    

    # Run on all input docs
    for docx_path in app_cfg.paths.list_input_docx():
        pipeline.run_on_file(docx_path, app_cfg)

    # Stop llama-server explicitly on normal shutdown
    type_print("Shutting down the server. Have a nice day!", color=Color.BLUE)
    server_proc = deps.get("llama-server")
    if server_proc is not None:
        server_proc.stop()
    


if __name__ == "__main__":
    main()
