

```yaml
@startuml Ecommerce Chatbot Workflow
title E-commerce Chatbot Workflow

start
:Initialize Chatbot
Setup Gemini & ChromaDB connections;

:Start Chat Loop;

repeat
    :Get User Input;
    
    if (input == "exit") then (yes)
        :End Session;
        stop
    else (no)
        :Send to Gemini AI Service;
        :Parse YAML Response;
        
        switch (Action Type)
        case (QUERY)
            :Extract Query Parameters;
            :**Query ChromaDB Database**
            (External Vector Database);
            :Perform Semantic Search;
            :Retrieve Product Data;
            :Send Results back to Gemini;
            :Parse Refined Response;
            
        case (DISPLAY)
            :Format & Display Results;
            if (needs_refinement?) then (yes)
                :Show Refinement Prompt
                (ask user for more details);
            endif
            
        case (SUMMARIZE)
            :Call Summarization Model;
            :Display Summary to User;
            
        endswitch
    endif
repeat while (continue chatting)

@enduml
```

### Generates PLantUML Code Flow

```yaml
@startuml Ecommerce Chatbot Sequence
title E-commerce Chatbot Sequence Diagram

actor User
participant Chatbot
participant Gemini
database ChromaDB as "ChromaDB\nVector Database"

== Initialization ==
Chatbot -> Gemini: Initialize connection
Chatbot -> ChromaDB: Initialize connection

== Chat Session ==
loop until user exits
    User -> Chatbot: Send query
    Chatbot -> Gemini: Process with conversation history
    Gemini --> Chatbot: YAML action response
    
    alt QUERY Action
        Chatbot -> ChromaDB: Semantic search query
        ChromaDB --> Chatbot: Vector search results
        Chatbot -> Gemini: Send RAG results for analysis
        Gemini --> Chatbot: Refined response
    else DISPLAY Action
        Chatbot -> User: Show results
        note right: May include refinement request
    else SUMMARIZE Action
        Chatbot -> Gemini: Summarization request
        Gemini --> Chatbot: Summary
        Chatbot -> User: Display summary
    end
    
    User -> Chatbot: Next message or "exit"
end

Chatbot -> User: Goodbye
@enduml
```