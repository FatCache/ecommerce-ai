import yaml
import argparse
from typing import Dict, List, Any, Optional, Protocol, Union
from gemini_config import configure_gemini
from text_utils import extract_yaml_from_markdown
from chroma_db_config import get_chromadb
from exceptions import ChatbotError, InvalidActionError, CollectionNotFoundError, GeminiAPIError
from models import ActionType, CollectionType, GeminiResponse, QueryParameters, DisplayParameters, SummarizeParameters, ChatbotConfig





# Protocol for action handlers (Strategy pattern)
class ActionHandler(Protocol):
    """Protocol for action handler methods."""

    def __call__(self, parameters: Dict[str, Any], user_input: str = "") -> None:
        """Handle an action with given parameters."""
        ...


# Configuration instance
config = ChatbotConfig()


def parse_yaml_response(gemini_response: str) -> GeminiResponse:
    """Parse YAML from Gemini response and return structured data."""
    cleaned_response = extract_yaml_from_markdown(gemini_response)

    try:
        raw_data = yaml.safe_load(cleaned_response)
    except yaml.YAMLError as e:
        # Fallback: Try to extract action from raw text
        return _fallback_parse_response(gemini_response)

    if not isinstance(raw_data, dict):
        return _fallback_parse_response(gemini_response)

    action_str = raw_data.get('action', '').strip()

    # If action is missing or empty, use fallback
    if not action_str:
        return _fallback_parse_response(gemini_response)

    try:
        action = ActionType(action_str.upper())  # Convert to uppercase and create enum
        parameters = raw_data.get('parameters', {})

        return GeminiResponse(action=action, parameters=parameters)
    except ValueError as e:
        # If enum creation fails, use fallback
        return _fallback_parse_response(gemini_response)


def _fallback_parse_response(gemini_response: str) -> GeminiResponse:
    """Fallback parsing when YAML structure is invalid."""
    # Simple fallback: assume it's a display action with the raw response as message
    return GeminiResponse(
        action=ActionType.DISPLAY,
        parameters={
            'message': f"I received your request but had trouble processing it. Here's what I got: {gemini_response[:200]}",
            'needs_refinement': True
        }
    )


def display_token_usage(usage_metadata: Any, label: str = "") -> None:
    """Display token usage information."""
    if usage_metadata:
        label_text = f" ({label})" if label else ""
        print(f"Token Usage{label_text}: Prompt={usage_metadata.prompt_token_count}, "
              f"Completion={usage_metadata.candidates_token_count}")


def display_results(message: str, data: Optional[List[Any]] = None, snippet_source: Optional[str] = None,
                   needs_refinement: bool = False) -> None:
    """Display chatbot response and handle refinement if needed."""
    print(f"\nChatbot: {message}")
    if data:
        for item in data:
            if isinstance(item, dict) and item.get('type') == 'snippet':
                source = item.get('source', snippet_source or 'RAG')
                print(f"  Snippet from {source}: \"{item.get('content')}\"")
            else:
                print(f"- {item}")

    if needs_refinement:
        # For iterative RAG
        if data:
            refinement_msg = ("Based on the initial search, I need more information to provide the best recommendation. "
                             "I found some options, but to narrow them down, could you specify your preferences?")
        else:
            refinement_msg = ("I couldn't find specific results matching your query. "
                             "Could you please rephrase or provide more details?")
        print(f"\nChatbot: {refinement_msg}")


class EcommerceChatbot:
    """E-commerce chatbot using Gemini and ChromaDB for RAG."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.main_model, self.summarization_model = configure_gemini()
        self.client, self.product_meta_collection, self.product_review_collection = get_chromadb()
        self.conversation = self.main_model.start_chat()

    def get_collection(self, collection_type: CollectionType) -> Any:
        """Get the appropriate collection based on enum type."""
        if collection_type == CollectionType.PRODUCT_META:
            return self.product_meta_collection
        elif collection_type == CollectionType.PRODUCT_REVIEW:
            return self.product_review_collection
        else:
            raise CollectionNotFoundError(f"Unknown collection '{collection_type.value}'")

    def handle_query_action(self, parameters: Dict[str, Any], user_input: str) -> None:
        """Handle QUERY action with RAG processing."""
        try:
            # Use dataclass for type safety
            query_params = QueryParameters(
                query_text=parameters.get('query_text', ''),
                collection=CollectionType(parameters.get('collection', '')),
                n_results=parameters.get('n_results', config.default_query_results)
            )
            collection = self.get_collection(query_params.collection)
        except (ValueError, CollectionNotFoundError) as e:
            print(f"Error: {e}")
            return

        print(f"\nQuerying ChromaDB for: '{query_params.query_text}' in '{query_params.collection.value}'\n")
        results = collection.query(query_texts=[query_params.query_text], n_results=query_params.n_results)

        # Send RAG results back to Gemini for processing
        rag_prompt = f"""
        Based on the user's last query and the following RAG results, please generate the next action (DISPLAY or SUMMARIZE).
        When using DISPLAY, always include at least one actual snippet from the RAG results in the data field.
        If results are insufficient, use DISPLAY with `needs_refinement: true`.

        Special handling for preference queries: Analyze RAG results to list key preferences without snippets.

        User's last query: "{query_params.query_text}"
        RAG Results: {results}

        Response MUST be in YAML format.
        """

        self.conversation.send_message(rag_prompt)
        gemini_response_after_rag = self.conversation.last.text
        if self.debug:
            print(f"\nGemini Response (after RAG):\n{gemini_response_after_rag}\n")

        try:
            response = parse_yaml_response(gemini_response_after_rag)
            action = response.action
            params = response.parameters

            # Use strategy pattern - map actions to handlers
            action_handlers = {
                ActionType.DISPLAY: self.handle_display_action,
                ActionType.SUMMARIZE: lambda p: self.handle_summarize_action(p, query_params.query_text),
            }

            handler = action_handlers.get(action)
            if handler:
                handler(params)
            else:
                raise InvalidActionError(f"Unknown action '{action.value}' after RAG processing.")

        except yaml.YAMLError as e:
            print("I'm sorry, I encountered an issue processing the information after a search. Please try rephrasing your request.")
        except Exception as e:
            print("I'm sorry, an unexpected error occurred while processing your request. Please try again.")

    def handle_display_action(self, parameters: Dict[str, Any]) -> None:
        """Handle DISPLAY action."""
        message = parameters.get('message', '')
        data = parameters.get('data')

        # Detect preference discovery responses - they list preferences and shouldn't trigger refinement
        is_preference_discovery = self._is_preference_discovery_response(message, data)

        needs_refinement = parameters.get('needs_refinement', False) and not is_preference_discovery

        display_params = DisplayParameters(
            message=message,
            data=data,
            snippet_source=parameters.get('snippet_source'),
            needs_refinement=needs_refinement
        )

        display_results(display_params.message, display_params.data,
                       display_params.snippet_source, display_params.needs_refinement)
        if self.debug:
            display_token_usage(self.conversation.last.usage_metadata, "DISPLAY")

    def handle_summarize_action(self, parameters: Dict[str, Any], user_input: str) -> None:
        """Handle SUMMARIZE action with enhanced comprehensive querying for 'tell me more' requests."""
        try:
            summarize_params = SummarizeParameters(
                text_to_summarize=parameters.get('text_to_summarize', '')
            )
        except ValueError:
            print("\nChatbot: I don't have any valid text to summarize. Please try rephrasing your request.")
            return

        if not summarize_params.text_to_summarize.strip():
            print("\nChatbot: I don't have any valid text to summarize. Please try rephrasing your request.")
            return

        # Use AI to classify if this request needs comprehensive information
        is_comprehensive_request = self._classify_comprehensive_request(summarize_params.text_to_summarize)

        if is_comprehensive_request:
            # For comprehensive requests, gather extensive data from both collections
            print("\nGathering comprehensive information for detailed summary...")

            try:
                # Query product metadata with broader results
                meta_results = self.product_meta_collection.query(
                    query_texts=[user_input],
                    n_results=config.comprehensive_meta_results
                )

                # Query product reviews for detailed feedback
                review_results = self.product_review_collection.query(
                    query_texts=[user_input],
                    n_results=config.comprehensive_review_results
                )

                # Validate query results structure
                if not isinstance(meta_results, dict) or not isinstance(review_results, dict):
                    if self.debug:
                        print(f"DEBUG: Invalid query results structure - meta: {type(meta_results)}, review: {type(review_results)}")
                    raise GeminiAPIError("Invalid query results structure")

                meta_count = len(meta_results.get('documents', []))
                review_count = len(review_results.get('documents', []))

                if self.debug:
                    print(f"DEBUG: Found {meta_count} meta results and {review_count} review results")
                    print(f"DEBUG: Meta results keys: {list(meta_results.keys())}")
                    print(f"DEBUG: Review results keys: {list(review_results.keys())}")

                # Check if we have any data
                if meta_count == 0 and review_count == 0:
                    if self.debug:
                        print("DEBUG: No data found, falling back to basic summarization")
                    print("\nChatbot: I couldn't find detailed information about that product. Let me provide a basic summary instead.")
                    # Fallback to regular summarization
                    summary_response = self.summarization_model.generate_content(summarize_params.text_to_summarize.strip())
                else:
                    # Combine data for concise, conversational summarization
                    comprehensive_data = f"""
Based on the user's request: "{summarize_params.text_to_summarize}"

Product Data ({meta_count} products, {review_count} reviews):
{meta_results}
{review_results}

Please provide a very brief, conversational summary in 3-4 sentences maximum that naturally answers the user's question. Focus on the most relevant insights and recommendations. Keep it concise and conversational, like you're chatting with a friend about products.
"""

                    # Limit the prompt size to avoid token limits (rough estimate)
                    max_length = 25000  # Conservative limit
                    if len(comprehensive_data) > max_length:
                        if self.debug:
                            print(f"DEBUG: Truncating prompt from {len(comprehensive_data)} to {max_length} characters")
                        comprehensive_data = comprehensive_data[:max_length] + "\n\n[Content truncated due to length]"

                print(f"\nGenerating concise summary from {meta_count} products and {review_count} reviews...")

                try:
                    summary_response = self.summarization_model.generate_content(comprehensive_data)

                    if self.debug:
                        print(f"DEBUG: Comprehensive summary generated successfully")

                except Exception as e:
                    if self.debug:
                        print(f"DEBUG: Comprehensive summarization failed: {e}")
                    raise GeminiAPIError(f"Failed to generate comprehensive summary: {e}") from e

            except GeminiAPIError as e:
                if self.debug:
                    print(f"DEBUG: Falling back to basic summarization due to: {e}")
                print(f"\nChatbot: I'm sorry, I encountered an issue gathering comprehensive data. Using basic summary instead.")
                # Fallback to basic summarization
                try:
                    summary_response = self.summarization_model.generate_content(summarize_params.text_to_summarize.strip())
                except Exception as fallback_error:
                    if self.debug:
                        print(f"DEBUG: Fallback summarization also failed: {fallback_error}")
                    print("\nChatbot: I'm sorry, I'm having trouble generating any summary right now. Please try again later.")
                    return

        else:
            # Standard summarization for regular cases
            print(f"\nSummarizing text using a cheaper model...")
            try:
                summary_response = self.summarization_model.generate_content(summarize_params.text_to_summarize.strip())
            except Exception as e:
                raise GeminiAPIError(f"Failed to generate summary: {e}") from e

        try:
            print(f"\nChatbot (Summary): {summary_response.text}")
            if self.debug:
                display_token_usage(summary_response.usage_metadata, "Summarization")
        except Exception as e:
            print(f"\nChatbot: I'm sorry, I encountered an issue while summarizing the text. It might be too long or contain unsupported content.")

    def _is_preference_discovery_response(self, message: str, data: Optional[List[Any]]) -> bool:
        """Detect if a DISPLAY response is a preference discovery list that shouldn't trigger refinement."""
        if not message:
            return False

        # Check for preference-related keywords in the message
        preference_keywords = [
            "preferences", "preference", "looking for", "would be helpful",
            "specific brand", "price range", "size", "color", "features",
            "target user", "material"
        ]

        message_lower = message.lower()
        has_preference_keywords = any(keyword in message_lower for keyword in preference_keywords)

        # Check if data contains preference objects (not product results)
        has_preference_data = False
        if data and isinstance(data, list):
            # Look for objects with 'preference' key (indicating preference list)
            has_preference_data = any(
                isinstance(item, dict) and 'preference' in item
                for item in data
            )

        return has_preference_keywords or has_preference_data

    def _classify_comprehensive_request(self, text: str) -> bool:
        """Use AI to classify if a summarization request needs comprehensive data gathering.

        This replaces hardcoded string matching with intelligent classification.
        """
        if not text or not text.strip():
            return False

        classification_prompt = f"""
Classify whether this user request requires comprehensive information gathering.
Respond with only "COMPREHENSIVE" or "STANDARD".

COMPREHENSIVE requests include:
- "tell me more about X"
- "what else can you tell me"
- "give me comprehensive information"
- "more details about X"
- "what more information do you have"
- Requests asking for extensive or detailed information

STANDARD requests include:
- "summarize this"
- "give me a quick overview"
- "brief summary"
- Regular summarization requests

User request: "{text.strip()}"
Classification:"""

        try:
            response = self.summarization_model.generate_content(classification_prompt)
            result = response.text.strip().upper()

            # Debug logging (only in debug mode)
            if self.debug:
                print(f"Classification result for '{text[:50]}...': {result}")

            return result == "COMPREHENSIVE"

        except Exception as e:
            # Fallback to simple string matching if AI classification fails
            if self.debug:
                print(f"AI classification failed ({e}), using fallback method")

            fallback_keywords = ["tell me more", "more about", "more information",
                               "comprehensive", "detailed", "extensive", "what else"]
            return any(keyword in text.lower() for keyword in fallback_keywords)

    def process_user_input(self, user_input: str) -> None:
        """Process a single user input and handle all responses internally."""
        for retry_count in range(config.max_retries):
            try:
                self.conversation.send_message(user_input)
                gemini_response = self.conversation.last.text

                if self.debug:
                    display_token_usage(self.conversation.last.usage_metadata)

                response = parse_yaml_response(gemini_response)
                action = response.action
                parameters = response.parameters

                # Use strategy pattern - map actions to handlers
                action_handlers: Dict[ActionType, ActionHandler] = {
                    ActionType.QUERY: lambda p: self.handle_query_action(p, user_input),
                    ActionType.DISPLAY: self.handle_display_action,
                    ActionType.SUMMARIZE: lambda p: self.handle_summarize_action(p, user_input),
                }

                handler = action_handlers.get(action)
                if handler:
                    handler(parameters)
                else:
                    raise InvalidActionError(f"Unknown action '{action.value}'")

                break  # Successfully processed, exit retry loop

            except (yaml.YAMLError, InvalidActionError) as e:
                print(f"Error parsing YAML response (Attempt {retry_count + 1}/{config.max_retries}): {e}")
                if retry_count == config.max_retries - 1:
                    print("Failed to get a proper YAML format after multiple retries.")
            except GeminiAPIError as e:
                print(f"I'm sorry, I encountered an API error: {e}")
                break  # Don't retry API errors
            except Exception as e:
                print("I'm sorry, an unexpected error occurred while processing your request. Please try again.")
                break  # Don't retry unexpected errors

    def start_chat(self) -> None:
        """Start the interactive chat session."""
        print("Welcome to the E-commerce Chatbot! How can I help you today? Type 'exit' to terminate session.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == config.exit_command:
                print("Goodbye!")
                break

            self.process_user_input(user_input)


def start_chat():
    """Main entry point for the chatbot."""
    parser = argparse.ArgumentParser(description="E-commerce AI Chatbot")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    chatbot = EcommerceChatbot(debug=args.debug)
    chatbot.start_chat()


if __name__ == "__main__":
    start_chat()