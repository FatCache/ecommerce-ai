# E-commerce AI Chatbot

## Project Overview

This project implements an AI-powered e-commerce chatbot designed to provide product recommendations and information based on user queries. The chatbot leverages Google Gemini for natural language understanding and response generation, and a Retrieval-Augmented Generation (RAG) system using ChromaDB for accessing product metadata and reviews.

The core idea is to structure the chatbot's responses in a YAML format, enabling it to perform specific actions like querying the RAG database, displaying information to the user, or summarizing retrieved data.

## Current Chatbot Functionality

The chatbot currently provides an interactive command-line interface where users can ask questions about e-commerce products. It uses the Gemini model to understand user intent and generate structured YAML responses. Based on these YAML instructions, the chatbot can:
*   **Query** a ChromaDB vector store for relevant product metadata or reviews.
*   **Display** information directly to the user, including product details and contextual snippets from customer reviews to explain positive experiences.
*   **Summarize** (currently prints a request for summarization) complex information, setting the stage for future advanced summarization capabilities.

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

*   **Interactive Chat Loop:** Engages users in a continuous conversation for product discovery.
*   **YAML Structured Responses:** Gemini generates responses in a structured YAML format, defining explicit actions and parameters.
    *   `QUERY`: Initiates a search on the ChromaDB RAG database.
    *   `DISPLAY`: Presents information or asks clarifying questions to the user.
    *   `SUMMARIZE`: (Placeholder) Intended for summarizing extensive RAG results.
*   **RAG Integration with ChromaDB:** Utilizes two ChromaDB collections:
    *   `product_meta`: For general product information (title, category, price, description).
    *   `product_review`: For detailed product reviews.
*   **Contextual Snippets:** When displaying product recommendations or positive experiences, the chatbot includes relevant snippets from raw RAG data to explain the rationale.
*   **Modular Configuration:** Gemini-specific configurations are externalized into `gemini_config.py` for better code organization and maintainability.
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

*   Implement full summarization logic using Gemini for the `SUMMARIZE` action.
*   Add more sophisticated query refinement mechanisms based on user feedback.
*   Integrate a more robust error handling and logging system.