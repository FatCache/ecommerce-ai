import chromadb
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
import yaml
from gemini_config import configure_gemini


def _extract_yaml_from_markdown(text):
    """Extracts YAML content from a markdown code block."""
    if text.strip().startswith('```yaml'):
        # Find the start and end of the YAML block
        start_index = text.find('```yaml') + len('```yaml')
        end_index = text.rfind('```')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            return text[start_index:end_index].strip()
    return text.strip()


def get_chromadb():
    chroma_db_name = 'chromadb_v1'
    client = chromadb.PersistentClient(path=f"./chromadbs/{chroma_db_name}")
    product_meta_collection = client.get_or_create_collection(
        name="product_meta",
        metadata={"description": "Product metadata collection"}
    )
    product_review_collection = client.get_or_create_collection(
        name="product_review",
        metadata={"description": "Product review collection"}
    )
    
    return client, product_meta_collection, product_review_collection

def start_chat():
    main_model, summarization_model = configure_gemini() # Get both models
    client, product_meta_collection, product_review_collection = get_chromadb()
    
    context_prompt = """
    You are an e-commerce chatbot. Your goal is to assist users with product recommendations and information.
    For each user input, you must determine the appropriate action: QUERY, DISPLAY, or SUMMARIZE.
    Your response MUST be structured in YAML format.

    **ChromaDB Collections and Fields:**
    You have access to two ChromaDB collections:

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

    **RAG Query Strategies:**
    When constructing a `QUERY` action, consider the following strategies to get good results from the vector database:
    *   **Semantic Relevance:** Use the user's query directly or rephrase it to capture the semantic meaning relevant to product attributes.
    *   **Field-Specific Queries:** If the user mentions specific attributes (e.g., "blue shirt under $50"), try to map these to relevant fields in the `product_meta` collection (e.g., `main_category`, `title`, `price`).
    *   **Leverage Reviews for Sentiment/Details:** Use the `product_review` collection to answer questions about product sentiment ("What do people say about this product?") or to find specific positive/negative feedback. You can use `parent_asin` to link reviews to products found in `product_meta`.
    *   **Iterative Refinement:** If initial results are not satisfactory or if the user's query is ambiguous, use the `DISPLAY` action with `needs_refinement: true` to ask clarifying questions. This allows for a more targeted subsequent query.

    Actions:
    - QUERY: Use this when you need to search the RAG database for product information.
      Parameters:
        query_text: The specific query string to use for the RAG search.
        collection: The collection to search (e.g., "product_meta", "product_review").
        n_results: The number of results to retrieve.

    Actions:
    - QUERY: Use this when you need to search the RAG database for product information.
      Parameters:
        query_text: The specific query string to use for the RAG search.
        collection: The collection to search (e.g., "product_meta", "product_review").
        n_results: The number of results to retrieve.
    - DISPLAY: Use this when you have information to show to the user, either from a RAG query or a direct answer.
      Parameters:
        message: The message to display to the user.
        data: (Optional) A list of items to present. Each item can be a dictionary with product details or a snippet from RAG.
        snippet_source: (Optional) If 'data' contains snippets, this field can indicate the source (e.g., "product_review").
        needs_refinement: (Optional) Boolean. Set to true if the chatbot needs more information from the user to refine a query.
    - SUMMARIZE: Use this when you need to summarize information, perhaps after a RAG query returns many results.
      Parameters:
        text_to_summarize: The text that needs to be summarized for the user.

    **Instructions for RAG Result Pre-processing and Iterative Querying:**

    1.  **After a QUERY action:** Once you have performed a QUERY and received results from the RAG database, you must analyze these results.
        *   If the results are directly useful and sufficient, generate a `DISPLAY` action with the relevant information, including snippets for positive experiences.
        *   If the results are too numerous or require condensation, generate a `SUMMARIZE` action.
        *   If the results are insufficient or ambiguous, and you need more specific input from the user to refine the query, generate a `DISPLAY` action with a clarifying `message` and set `needs_refinement: true`.

    2.  **When providing recommendations or information about positive experiences:** Include a relevant snippet from the raw data retrieved from the vector database in the 'data' field of the DISPLAY action. The snippet should explain *why* people have a positive experience.

    Example YAML response for QUERY:
    ```yaml
    action: QUERY
    parameters:
      query_text: "compression sleeves for running"
      collection: "product_meta"
      n_results: 5
    ```

    Example YAML response for DISPLAY with product details:
    ```yaml
    action: DISPLAY
    parameters:
      message: "Here are some compression sleeves for running:"
      data:
        - product_name: "Product A"
          price: "$25.99"
        - product_name: "Product B"
          price: "$29.99"
    ```

    Example YAML response for DISPLAY with a positive experience snippet:
    ```yaml
    action: DISPLAY
    parameters:
      message: "People have a positive experience with Product X because it offers great comfort and durability."
      data:
        - type: "snippet"
          content: "I've been using these sleeves for months, and they're incredibly comfortable during long runs. The material holds up really well after many washes."
          source: "product_review"
    ```

    Example YAML response for DISPLAY requesting refinement:
    ```yaml
    action: DISPLAY
    parameters:
      message: "I found several options, but to give you the best recommendation, could you tell me more about your preferences? For example, are you looking for a specific brand, price range, or material?"
      needs_refinement: true
    ```

    Example YAML response for SUMMARIZE:
    ```yaml
    action: SUMMARIZE
    parameters:
      text_to_summarize: "The user is looking for running gear, specifically compression sleeves. The RAG query returned several options, which need to be summarized for clarity."
    ```

    Always strive to provide the most helpful and concise response.
    """
    convo = main_model.start_chat(history=[]) # Use main_model for the conversation

    print("Welcome to the E-commerce Chatbot! How can I help you today? Type 'exit' to terminate session.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        full_prompt = f"{context_prompt}\nUser: {user_input}\n"
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                convo.send_message(full_prompt)
                gemini_response = convo.last.text
                print(f"\nGemini Response:\n{gemini_response}\n")

                # Display token usage for this turn
                if convo.last.usage_metadata:
                    print(f"Token Usage: Prompt={convo.last.usage_metadata.prompt_token_count}, Completion={convo.last.usage_metadata.candidates_token_count}")

                # Parse YAML response and take action
                cleaned_gemini_response = _extract_yaml_from_markdown(gemini_response)
                response_yaml = yaml.safe_load(cleaned_gemini_response)
                action = response_yaml.get('action')
                parameters = response_yaml.get('parameters', {})

                if action == 'QUERY':
                    query_text = parameters.get('query_text')
                    collection_name = parameters.get('collection')
                    n_results = parameters.get('n_results', 5)

                    if collection_name == "product_meta":
                        collection = product_meta_collection
                    elif collection_name == "product_review":
                        collection = product_review_collection
                    else:
                        print(f"Error: Unknown collection '{collection_name}'")
                        break # Exit retry loop on unknown collection

                    print(f"\nQuerying ChromaDB for: '{query_text}' in '{collection_name}'\n")
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=n_results
                    )
                    print("Query Results:")
                    print(results)
                    
                    # Now, send the RAG results back to Gemini for pre-processing and decision making
                    # This is the core of "Pre-processing RAG Results" and "Iterative RAG"
                    rag_response_prompt = f"""
                    Based on the user's last query and the following RAG results, please generate the next action (DISPLAY or SUMMARIZE).
                    If the results are insufficient or require more user input for refinement, use the DISPLAY action with `needs_refinement: true`.

                    User's last query: "{user_input}"
                    RAG Results: {results}

                    Your response MUST be in YAML format, following the defined actions.
                    """
                    convo.send_message(rag_response_prompt)
                    gemini_response_after_rag = convo.last.text
                    print(f"\nGemini Response (after RAG):\n{gemini_response_after_rag}\n")

                    # Attempt to parse this new Gemini response
                    try:
                        cleaned_gemini_response_after_rag = _extract_yaml_from_markdown(gemini_response_after_rag)
                        response_yaml_after_rag = yaml.safe_load(cleaned_gemini_response_after_rag)
                        action_after_rag = response_yaml_after_rag.get('action')
                        parameters_after_rag = response_yaml_after_rag.get('parameters', {})

                        if action_after_rag == 'DISPLAY':
                            message = parameters_after_rag.get('message')
                            data = parameters_after_rag.get('data')
                            snippet_source = parameters_after_rag.get('snippet_source')
                            needs_refinement = parameters_after_rag.get('needs_refinement', False)

                            print(f"\nChatbot: {message}")
                            if data:
                                for item in data:
                                    if isinstance(item, dict) and item.get('type') == 'snippet':
                                        print(f"  Snippet from {item.get('source', 'RAG')}: \"{item.get('content')}\"")
                                    else:
                                        print(f"- {item}")
                            
                            if needs_refinement:
                                # This triggers the iterative RAG by asking for more input
                                # Providing context from RAG results to help user refine their search
                                refinement_context = "Based on the initial search, I need more information to provide the best recommendation. "
                                if data:
                                    # If there's data, try to extract a relevant snippet or mention product attributes
                                    # For simplicity, we'll just mention that results were found but need refinement.
                                    refinement_context += "I found some options, but to narrow them down, could you specify your preferences?"
                                else:
                                    refinement_context += "I couldn't find specific results matching your query. Could you please rephrase or provide more details?"
                                
                                print(f"\nChatbot: {refinement_context}")
                                # The loop will continue, and the next user_input will be used for a new query.
                            
                            # Display token usage for the DISPLAY action after RAG processing
                            if convo.last.usage_metadata:
                                print(f"Token Usage (Post-RAG DISPLAY): Prompt={convo.last.usage_metadata.prompt_token_count}, Completion={convo.last.usage_metadata.candidates_token_count}")

                            # Removed break here to allow the outer loop to continue and prompt for user input for refinement.
                            # break # Successfully processed, exit retry loop

                        elif action_after_rag == 'SUMMARIZE':
                            text_to_summarize = parameters_after_rag.get('text_to_summarize')
                            print(f"\nChatbot (Summary Request): {text_to_summarize}")
                            break # Successfully processed, exit retry loop
                        else:
                            print(f"Error: Unknown action '{action_after_rag}' after RAG processing.")
                            break # Exit retry loop on unknown action after RAG

                    except yaml.YAMLError as ye_rag:
                        print(f"Error parsing YAML response after RAG: {ye_rag}")
                        print(f"Raw Gemini Response (after RAG):\n{gemini_response_after_rag}")
                        # This specific error should not trigger a full retry loop for the *initial* query,
                        # but rather indicate an issue with Gemini's response post-RAG.
                        # For now, we'll just break. A more robust solution might involve a separate retry for this step.
                        break
                    except Exception as ex_rag:
                        print(f"Error processing Gemini response after RAG: {ex_rag}")
                        break

                elif action == 'DISPLAY':
                    message = parameters.get('message')
                    data = parameters.get('data')
                    snippet_source = parameters.get('snippet_source')
                    needs_refinement = parameters.get('needs_refinement', False)

                    print(f"\nChatbot: {message}")
                    if data:
                        for item in data:
                            if isinstance(item, dict) and item.get('type') == 'snippet':
                                print(f"  Snippet from {item.get('source', 'RAG')}: \"{item.get('content')}\"")
                            else:
                                print(f"- {item}")
                    
                    if needs_refinement:
                        print("\nChatbot: Please provide more details to refine your search.")
                    
                    # Display token usage for the DISPLAY action
                    if convo.last.usage_metadata:
                        print(f"Token Usage (DISPLAY): Prompt={convo.last.usage_metadata.prompt_token_count}, Completion={convo.last.usage_metadata.candidates_token_count}")

                    # Removed break here to allow the outer loop to continue and prompt for user input for refinement.
                    # break # Successfully processed, exit retry loop

                elif action == 'SUMMARIZE':
                    text_to_summarize = parameters.get('text_to_summarize')
                    # Use the cheaper summarization model for this task
                    print(f"\nSummarizing text using a cheaper model...")
                    summary_response = summarization_model.generate_content(text_to_summarize)
                    print(f"\nChatbot (Summary): {summary_response.text}")
                    
                    # Display token usage for the summarization
                    if summary_response.usage_metadata:
                        print(f"Token Usage (Summarization): Prompt={summary_response.usage_metadata.prompt_token_count}, Completion={summary_response.usage_metadata.candidates_token_count}")

                    break # Successfully processed, exit retry loop

                else:
                    print(f"Error: Unknown action '{action}'")
                    break # Exit retry loop on unknown action

            except yaml.YAMLError as ye:
                retry_count += 1
                print(f"Error parsing YAML response (Attempt {retry_count}/{max_retries}): {ye}")
                print(f"Raw Gemini Response:\n{gemini_response}")
                if retry_count == max_retries:
                    print("Failed to get a proper YAML format after multiple retries. Something went wrong.")
                    break # Exit chat loop entirely
                # Optionally, you could modify the prompt here to tell Gemini to be more careful with YAML.
                # For simplicity, we're just retrying with the same prompt.
            except Exception as ex:
                print(f"Error processing Gemini response: {ex}")
                break # Exit retry loop on other exceptions
        
        if retry_count == max_retries:
            print("Exiting chat due to persistent YAML parsing errors.")
            break # Exit the main chat loop if max retries reached


if __name__ == "__main__":
    start_chat()