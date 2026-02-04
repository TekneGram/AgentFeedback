graph TD
    %% Styling
    classDef config fill:#f9f,stroke:#333,stroke-width:2px;
    classDef core fill:#bbf,stroke:#333,stroke-width:2px;
    classDef nlp fill:#dfd,stroke:#333,stroke-width:1px;
    classDef io fill:#ffd,stroke:#333,stroke-width:1px;

    %% Level 1: Initialization
    subgraph Initialization
        settings[app.settings] -- "AppConfig Object" --> container[app.container]
    end

    %% Level 2: Orchestration
    container -- "Injected Instances" --> pipeline[app.pipeline]

    %% Level 3: Data Ingest & Prep
    subgraph Data_Prep [Input & Text Processing]
        pipeline -- "File Path" --> docx_loader[inout.docx_loader]
        docx_loader -- "Paragraph List" --> pipeline
        pipeline -- "Raw Text" --> text_header[text.header_extractor]
        pipeline -- "Text Blocks" --> text_split[text.sentence_splitter]
    end

    %% Level 4: Analysis Services
    subgraph Analysis_Services [Analysis Engine]
        pipeline -- "Sentences" --> ged_service[services.ged_service]
        ged_service -- "Tokens" --> ged_model[nlp.ged_bert]
        ged_model -- "Error Tags" --> ged_service

        pipeline -- "Prompt Context" --> llm_service[services.llm_service]
        
        subgraph LLM_Tasks [LLM Task Suite]
            llm_service -- "Task Logic" --> meta_task[nlp.llm.tasks.metadata_ext]
            llm_service -- "Task Logic" --> paragraph_task[nlp.llm.tasks.para_analysis]
            llm_service -- "Task Logic" --> grammar_task[nlp.llm.tasks.grammar_corr]
        end

        llm_service -- "OpenAI Schema" --> llm_client[nlp.llm.client]
        llm_client -- "HTTP / JSON" --> llm_server[nlp.llm.server_process]
    end

    %% Level 5: Logging & Output
    subgraph Output_Generation [Logging & Output]
        pipeline -- "Trace Dict" --> explain_rec[services.explainability]
        explain_rec -- "Log Strings" --> explain_writer[inout.explainability_writer]

        pipeline -- "Corrected Text" --> docx_out[services.docx_output_service]
        docx_out -- "XML/DOM Ops" --> track_changes[docx_tools.track_changes_editor]
    end

    %% Assign Classes
    class settings,container config;
    class pipeline core;
    class ged_model,meta_task,paragraph_task,grammar_task,llm_client,llm_server nlp;
    class docx_loader,docx_out,explain_writer io;