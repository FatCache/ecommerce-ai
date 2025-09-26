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
    print("Models configured and ChromaDB initialized.")
    
    return client, product_meta_collection, product_review_collection

def start_chat():
    main_model, summarization_model = configure_gemini() # Get both models
    client, product_meta_collection, product_review_collection = get_chromadb()

    convo = main_model.start_chat() # Use main_model for the conversation

    print("Welcome to the E-commerce Chatbot! How can I help you today? Type 'exit' to terminate session.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        full_prompt = user_input
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                convo.send_message(full_prompt)
                gemini_response = convo.last.text
                #print(f"\nGemini Response:\n{gemini_response}\n")

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
                    results = collection.query(query_texts=[query_text],n_results=n_results)
                    ##print("Query Results:")
                    ##print(results)
                    
                    # Now, send the RAG results back to Gemini for pre-processing and decision making
                    # This is the core of "Pre-processing RAG Results" and "Iterative RAG"
                    rag_response_prompt = f"""
                    Based on the user's last query and the following RAG results, please generate the next action (DISPLAY or SUMMARIZE).
                    When using DISPLAY, always include at least one actual snippet from the RAG results in the data field to show real customer feedback or product details.
                    If the results are insufficient or require more user input for refinement, use the DISPLAY action with `needs_refinement: true`.

                    **Special handling for preference discovery queries:** If the user's query is asking what preferences or information is needed for a product (e.g., "what preferences do you need for tennis shoes"), analyze the RAG results to identify common product attributes and features. Generate a DISPLAY action with a helpful message listing key preferences (such as size, brand, price range, features) based on the available product data. Do not include raw RAG snippets in the response; synthesize a user-friendly list of preferences.

                    User's last query: "{user_input}"
                    RAG Results: {results}

                    Your response MUST be in YAML format, following the defined actions.
                    """
                    convo.send_message(rag_response_prompt)
                    gemini_response_after_rag = convo.last.text
                    #print(f"\nGemini Response (after RAG):\n{gemini_response_after_rag}\n")

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
                                break  # Exit retry loop on refinement
                            else:
                                # Successfully displayed results without needing refinement, exit retry loop
                                break

                            # Display token usage for the DISPLAY action after RAG processing
                            if convo.last.usage_metadata:
                                print(f"Token Usage (Post-RAG DISPLAY): Prompt={convo.last.usage_metadata.prompt_token_count}, Completion={convo.last.usage_metadata.candidates_token_count}")


                        elif action_after_rag == 'SUMMARIZE':
                            text_to_summarize = parameters_after_rag.get('text_to_summarize')
                            print(f"\nChatbot (Summary Request): {text_to_summarize}")
                            break # Successfully processed, exit retry loop
                        else:
                            print(f"Error: Unknown action '{action_after_rag}' after RAG processing.")
                            break # Exit retry loop on unknown action after RAG

                    except yaml.YAMLError as ye_rag:
                        print(f"I'm sorry, I encountered an issue processing the information after a search. Please try rephrasing your request.")
                        print(f"Internal error details: {ye_rag}") # Keep internal details for debugging if needed, but prioritize user message
                        # print(f"Raw Gemini Response (after RAG):\n{gemini_response_after_rag}") # Optionally keep this for debugging
                        break
                    except Exception as ex_rag:
                        print(f"I'm sorry, an unexpected error occurred while processing your request. Please try again.")
                        print(f"Internal error details: {ex_rag}") # Optionally keep for debugging
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
                        break  # Exit retry loop on refinement

                    # Display token usage for the DISPLAY action
                    if convo.last.usage_metadata:
                        print(f"Token Usage (DISPLAY): Prompt={convo.last.usage_metadata.prompt_token_count}, Completion={convo.last.usage_metadata.candidates_token_count}")


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
                print(f"I'm sorry, an unexpected error occurred while processing your request. Please try again.")
                print(f"Internal error details: {ex}") # Optionally keep for debugging
                break # Exit retry loop on other exceptions
        
        if retry_count == max_retries:
            print("I'm sorry, I'm having trouble processing your request right now. Please try again later.")
            # Continue to next user input instead of exiting


if __name__ == "__main__":
    start_chat()
