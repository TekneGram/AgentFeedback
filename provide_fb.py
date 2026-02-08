from app.settings import build_settings
from app.container import build_container
from app.pipeline import FeedbackPipeline
from app.llama_bootstrap import bootstrap_llama
from app.model_selection import select_model_and_update_config

from utils.terminal_ui import Color, type_print, stage

def main():
    # Build config (paths/run/ged/llama)
    type_print("Building the app settings", color=Color.BLUE)
    app_cfg = build_settings()

    # Choose a model based on hardware and user preference
    type_print("Selecting the best model for your system", color=Color.BLUE)
    app_cfg = select_model_and_update_config(app_cfg)

    # Ensure gguf + llama-server exist, and return updated cfg (paths resolved)
    type_print("Bootstrapping Llama", color=Color.BLUE)
    app_cfg = bootstrap_llama(app_cfg)
    type_print("Configuration complete:\n------------------")
    type_print(f"Language Model: {app_cfg.llama.llama_model_display_name} (Set in llama config)\n", color=Color.BLUE)
    type_print(f"Language Model Family: {app_cfg.llama.llama_model_family} (Set in llama config)\n", color=Color.BLUE)
    type_print(f"Server url: {app_cfg.llama.llama_server_url} (Set in llama config)\n", color=Color.BLUE)
    type_print(f"Multi-modal projected used: {app_cfg.llama.hf_mmproj_filename} (Set in llama config)\n", color=Color.BLUE)
    type_print(f"Grammer Error Detection: {app_cfg.ged.model_name}, batch size: {app_cfg.ged.batch_size} (Set in GED config)\n", color=Color.BLUE)
    type_print(f"Maximum LLM GED corrections: {app_cfg.run.max_llm_corrections} (Set in run config)\n", color=Color.BLUE)
    type_print(f"Your grading input folder: {app_cfg.paths.input_docx_folder} (Set in paths config)\n", color=Color.BLUE)
    type_print(f"Your grading completed folder: {app_cfg.paths.output_docx_folder}(Set in paths config)\n", color=Color.BLUE)
    type_print(f"Your grading explained folder: {app_cfg.paths.explained_txt_folder} (Set in paths config)\n", color=Color.BLUE)
    type_print(f"Mode: {'Single Paragraph' if app_cfg.run.single_paragraph_mode else 'Essay'} (Set in run config)\n", color=Color.BLUE)
    type_print(f"Word document author name: {app_cfg.run.author} (Set in run config) \n", color=Color.BLUE)

    # Build all services/objects via a container
    type_print("Loading a large language model. This will take a large amount of your system's memory. Closing unused apps and browser windows can help.", color=Color.RED)
    with stage("Building all the services and loading a large language model.", color=Color.BLUE):
        deps = build_container(app_cfg)

    # Construct the pipeline and inject the dependencies as kwargs (named arguments)
    # pipeline = FeedbackPipeline(**deps)
    type_print("Constructing the nlp pipeline.", color=Color.BLUE)

    # Inject the dependencies (Dependency injection)
    # Keep coupling with configuration low by passing app_cfg into an object method later.
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
