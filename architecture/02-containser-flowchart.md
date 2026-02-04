graph TD
    %% Styling
    classDef actor fill:#f5f5f5,stroke:#333,stroke-dasharray: 5 5;
    classDef internal fill:#d4ebf2,stroke:#0e5a71,stroke-width:2px;
    classDef external fill:#f9e4b7,stroke:#8a6d3b,stroke-width:2px;
    classDef storage fill:#dbfad6,stroke:#3b7a30,stroke-width:2px;

    %% Level 0: The User
    User((Teacher or TA)) -- "1. Runs" --> CLI[CLI Runner]

    %% Level 1: Initialization & External Resources
    subgraph Setup [Environment Setup]
        CLI -- "2a. Download if missing" --> HF[Hugging Face Hub]
        CLI -- "2b. Read Model" --> Models[(Model Store)]
    end

    %% Level 2: The Core Hub
    CLI -- "3. Invokes" --> Pipe[Feedback Pipeline]

    %% Level 3: Processing Services (Arranged to prevent cross-over)
    subgraph Processing [Internal Processing]
        direction LR
        DOCX_L[DOCX Loader] <-- "Paragraphs" --> Pipe
        GED[GED Service] <-- "Scores" --> Pipe
        LLM_S[LLM Service] <-- "Analysis" --> Pipe
    end

    %% Level 4: External Inference
    LLM_S -- "HTTP Requests" --> Llama[llama-server]
    Llama -- "Load GGUF" --> Models

    %% Level 5: Output Handling
    subgraph Output [Persistence Layer]
        Pipe -- "Tracked Edits" --> DOCX_O[DOCX Output Service]
        Pipe -- "Logs" --> EXP[Explainability Service]
        
        EXP --> EXP_W[Explainability Writer]
        
        EXP_W --> DB_EXP[(Explainability Output)]
        DOCX_O --> DB_DOCX[(DOCX Output Files)]
    end

    %% Assign Classes
    class User actor;
    class CLI,Pipe,DOCX_L,GED,LLM_S,DOCX_O,EXP,EXP_W internal;
    class HF,Llama external;
    class Models,DB_EXP,DB_DOCX storage;