# E-commerce AI Chatbot

## Project Overview

This project implements an AI-powered e-commerce chatbot designed to provide product recommendations and information based on user queries. The chatbot leverages Google Gemini for natural language understanding and response generation, and a Retrieval-Augmented Generation (RAG) system using ChromaDB for accessing product metadata and reviews.

The chatbot uses Gemini's `system_instruction` to define its role and behavior, and relies on the model's built-in conversation history management to maintain context across interactions without manual prompt concatenation. Response generation follows a structured YAML format, enabling explicit actions like querying the RAG database, displaying information to the user, or summarizing retrieved data.

## Current Chatbot Functionality

The chatbot provides an interactive command-line interface where users can engage in natural conversations about e-commerce products. It leverages Gemini's chat session capabilities to automatically maintain conversation history, eliminating the need to manually pass prior context with each prompt.

Using a detailed `system_instruction` prompt ([`prompts.py`](prompts.py)), Gemini generates structured YAML responses to determine appropriate actions. Based on these instructions, the chatbot can:
*   **Query** ChromaDB vector stores for relevant product metadata or reviews.
*   **Display** information to the user, including product details and contextual snippets from customer reviews.
*   **Summarize** (currently prints a request for summarization) complex information, with potential for advanced summarization.

## Enhanced RAG Query Strategies

The chatbot's prompt has been enhanced to guide Gemini in generating more effective RAG queries by providing detailed information about the available ChromaDB collections and their fields, along with specific strategies:

*   **Collection Awareness:** Gemini is explicitly informed about the two collections (`product_meta` and `product_review`) and their respective fields.
*   **Semantic Relevance:** Queries are designed to capture the semantic meaning of user requests, mapping them to product attributes.
*   **Field-Specific Queries:** Gemini can now leverage specific fields like `main_category`, `title`, `price`, and `average_rating` from `product_meta` for targeted searches.
*   **Review Analysis:** The `product_review` collection is used for sentiment analysis and extracting specific feedback, linked via `parent_asin`.
*   **Iterative Refinement:** The chatbot supports iterative RAG by allowing Gemini to request clarifying questions (`needs_refinement: true`) when initial results are insufficient. When this occurs, the chatbot will prompt the user for more details, enabling a more targeted subsequent query. The chatbot now provides more context in its refinement prompts, helping users understand what information is needed to improve the search results.

## ChromaDB Collections

This project utilizes exactly two collections within ChromaDB for storing and retrieving product-related data:

1.  **`product_meta` Collection:**
    *   **Purpose:** Stores core metadata for each product.
    *   **Fields:**
        *   `main_category`: The primary category of the product (e.g., "Apparel", "Electronics").
        *   `title`: The name of the product.
        *   `average_rating`: The average customer rating for the product.
        *   `feature`: Key features of the product.
        *   `description`: A detailed description of the product.
        *   `price`: The price of the product.

2.  **`product_review` Collection:**
    *   **Purpose:** Stores customer reviews for products, enabling sentiment analysis and detailed feedback.
    *   **Fields:** (Specific fields may vary based on data ingestion, but typically include)
        *   `review_text`: The content of the customer review.
        *   `rating`: The rating given by the customer.
        *   `parent_asin`: An identifier linking the review to its product in the `product_meta` collection.
        *   `review_date`: The date the review was posted.

**Note:** The system is designed to strictly use only these two collections for RAG operations.

## Features

*   **Conversation History Management:** Gemini's chat session automatically maintains context across interactions, allowing natural, multi-turn conversations without manual history concatenation.
*   **System Instruction Guidance:** Uses a comprehensive `system_instruction` ([`prompts.py`](prompts.py)) to define chatbot behavior, available actions, and RAG strategies.
*   **Interactive Chat Loop:** Engages users in a continuous conversation for product discovery.
*   **YAML Structured Responses:** Gemini generates responses in a structured YAML format, defining explicit actions and parameters:
    *   `QUERY`: Initiates a search on the ChromaDB RAG database.
    *   `DISPLAY`: Presents information or asks clarifying questions to the user.
    *   `SUMMARIZE`: (Placeholder) Intended for summarizing extensive RAG results.
*   **RAG Integration with ChromaDB:** Utilizes two ChromaDB collections:
    *   `product_meta`: For general product information (title, category, price, description).
    *   `product_review`: For detailed product reviews.
*   **Contextual Snippets:** When displaying product recommendations or positive experiences, the chatbot includes relevant snippets from raw RAG data to explain the rationale.
*   **Iterative RAG:** Supports query refinement by prompting users for additional details when initial results are insufficient.
*   **Modular Configuration:** Gemini-specific configurations (including generation parameters, safety settings, and system instruction) are externalized into `gemini_config.py`.
*   **Token Usage Display:** Provides insights into resource consumption by displaying prompt and completion token counts for each Gemini response.
*   **Cost-Optimized Summarization:** Employs a cheaper Gemini model for summarization tasks.
*   **Robust Error Handling:** Includes a retry mechanism for YAML parsing failures and graceful exit on persistent errors.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/ecommerce-ai.git
    cd ecommerce-ai
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Google Gemini API Key:**
    Open `ecommerce-ai/gemini_config.py` and replace `'AIzaSyCvN4cbmBn6969Q8jaSAyCIuw0P_yVi8WU'` with your actual Google Gemini API key.
    ```python
    genai.configure(api_key='YOUR_GOOGLE_API_KEY')
    ```
    *(Note: It's recommended to use environment variables for API keys in production environments.)*

5.  **Prepare ChromaDB (if not already done):**
    Ensure your ChromaDB collections (`product_meta` and `product_review`) are populated with data. Refer to `data-processor-exp.py` or `dataset-processor.py` for data ingestion examples. The database files are expected to be in `./chromadbs/chromadb_v1/`.

## How to Run the Chatbot

To start the chatbot, execute the `chatbot.py` script:

```bash
python ecommerce-ai/chatbot.py
```

The chatbot will greet you, and you can start typing your queries. Type `exit` to end the chat.

## Project Structure

```
ecommerce-ai/
├── chatbot.py              # Main chatbot logic, handles user interaction and action execution.
├── data-processor-exp.py   # Example script for processing and ingesting data into ChromaDB.
├── dataset-processor.py    # Another script for dataset processing.
├── gemini_config.py        # Externalized Google Gemini API configuration and model setup.
├── main.py                 # (Potentially) Main application entry point or other utility.
├── README.md               # Project documentation.
├── requirements.txt        # Python dependencies.
├── server.py               # (Potentially) A FastAPI server for API endpoints.
├── .venv/                  # Python virtual environment.
└── chromadbs/
    └── chromadb_v1/        # Persistent ChromaDB storage.
        └── ...             # ChromaDB data files.
```

## Future Enhancements / TODOs

*   use Gemini `tool` to get rid of YAML way to call internal functions
*   implement 'producer consumer' style to rebuilt chroma database