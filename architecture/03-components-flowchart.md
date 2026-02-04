graph LR
    %% Styling
    classDef runner fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef pipe fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef llm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef ext fill:#eceff1,stroke:#263238,stroke-dasharray: 5 5;

    %% Level 1: CLI Runner (The Orchestrator)
    subgraph CLI_Runner [CLI Runner]
        user((Teacher)) --> provide[provide_fb]
        provide --> settings[Settings Builder]
        provide --> bootstrap[Llama Bootstrap]
        provide --> container[Container Builder]
    end

    %% Level 2: Setup to Execution
    container --> server[Llama Server Process]
    container --> pipeline_core[Feedback Pipeline Core]

    %% Level 3: The Pipeline Core & Sub-components
    subgraph Pipeline_Engine [Pipeline Engine]
        pipeline_core --> docx[Docx Loader]
        pipeline_core --> ged[GED Service]
        pipeline_core --> llm_svc[LLM Service]
        pipeline_core --> explain[Explainability Recorder]
        pipeline_core --> docx_out[DOCX Output Service]
    end

    %% Level 4: LLM Client Layer
    subgraph LLM_Client_Layer [LLM Client Layer]
        llm_svc --> tasks[LLM Tasks]
        tasks --> client[OpenAI Client]
    end

    %% Level 5: External Process & Storage
    client -- "HTTP" --> llama_bin[[llama-server]]
    server -- "Manage" --> llama_bin
    bootstrap -.-> models[(Model Store)]
    llama_bin -- "Load" --> models

    %% Logging Sub-flow (Branching down to avoid overlap)
    explain --> explain_writer[Explainability Writer]

    %% Assign Classes
    class provide,settings,bootstrap,container runner;
    class pipeline_core,docx,ged,llm_svc,explain,docx_out,explain_writer pipe;
    class tasks,client,server llm;
    class llama_bin ext;