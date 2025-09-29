import pytest
from unittest.mock import patch, MagicMock
from io import StringIO


class TestEcommerceChatbot:
    """Test suite for the EcommerceChatbot class."""

    def test_chatbot_initialization(self):
        """Test that chatbot can be initialized without errors."""
        from chatbot import EcommerceChatbot

        # Mock the dependencies
        mock_main_model = MagicMock()
        mock_summary_model = MagicMock()
        mock_client = MagicMock()
        mock_meta_col = MagicMock()
        mock_review_col = MagicMock()

        with patch('chatbot.configure_gemini', return_value=(mock_main_model, mock_summary_model)), \
             patch('chatbot.get_chromadb', return_value=(mock_client, mock_meta_col, mock_review_col)):

            chatbot = EcommerceChatbot()
            assert chatbot.main_model == mock_main_model
            assert chatbot.summarization_model == mock_summary_model
            assert chatbot.product_meta_collection == mock_meta_col
            assert chatbot.product_review_collection == mock_review_col

    def test_debug_flag_initialization(self):
        """Test that debug flag is properly set during initialization."""
        from chatbot import EcommerceChatbot

        mock_main_model = MagicMock()
        mock_summary_model = MagicMock()
        mock_client = MagicMock()
        mock_meta_col = MagicMock()
        mock_review_col = MagicMock()

        with patch('chatbot.configure_gemini', return_value=(mock_main_model, mock_summary_model)), \
             patch('chatbot.get_chromadb', return_value=(mock_client, mock_meta_col, mock_review_col)):

            chatbot_debug = EcommerceChatbot(debug=True)
            assert chatbot_debug.debug is True

            chatbot_no_debug = EcommerceChatbot(debug=False)
            assert chatbot_no_debug.debug is False

            chatbot_default = EcommerceChatbot()
            assert chatbot_default.debug is False

    def test_get_collection_returns_correct_collections(self):
        """Test that get_collection returns the correct collection objects."""
        from chatbot import EcommerceChatbot, CollectionType

        mock_main_model = MagicMock()
        mock_summary_model = MagicMock()
        mock_client = MagicMock()
        mock_meta_col = MagicMock()
        mock_review_col = MagicMock()

        with patch('chatbot.configure_gemini', return_value=(mock_main_model, mock_summary_model)), \
             patch('chatbot.get_chromadb', return_value=(mock_client, mock_meta_col, mock_review_col)):

            chatbot = EcommerceChatbot()

            assert chatbot.get_collection(CollectionType.PRODUCT_META) == mock_meta_col
            assert chatbot.get_collection(CollectionType.PRODUCT_REVIEW) == mock_review_col

    def test_get_collection_raises_error_for_unknown_collection(self):
        """Test that get_collection raises CollectionNotFoundError for unknown collection types."""
        from chatbot import EcommerceChatbot, CollectionType, CollectionNotFoundError

        mock_main_model = MagicMock()
        mock_summary_model = MagicMock()
        mock_client = MagicMock()
        mock_meta_col = MagicMock()
        mock_review_col = MagicMock()

        with patch('chatbot.configure_gemini', return_value=(mock_main_model, mock_summary_model)), \
             patch('chatbot.get_chromadb', return_value=(mock_client, mock_meta_col, mock_review_col)):

            chatbot = EcommerceChatbot()

            # Create a mock enum value that's not in our defined enums
            mock_unknown_collection = MagicMock()
            mock_unknown_collection.value = "unknown_collection"

            with pytest.raises(CollectionNotFoundError, match="Unknown collection 'unknown_collection'"):
                chatbot.get_collection(mock_unknown_collection)


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_parse_yaml_response(self):
        """Test YAML parsing functionality."""
        from chatbot import parse_yaml_response, ActionType

        yaml_text = """
action: DISPLAY
parameters:
  message: "Test message"
  data:
    - item1
    - item2
"""

        result = parse_yaml_response(yaml_text)
        assert result.action == ActionType.DISPLAY
        assert result.parameters["message"] == "Test message"
        assert len(result.parameters["data"]) == 2

    def test_display_results_formats_output(self, capsys):
        """Test that display_results formats output correctly."""
        from chatbot import display_results

        # Test basic message display
        display_results("Test message")
        captured = capsys.readouterr()
        assert "Chatbot: Test message" in captured.out

    def test_display_results_handles_data_list(self, capsys):
        """Test that display_results handles data lists correctly."""
        from chatbot import display_results

        test_data = ["item1", "item2", {"type": "snippet", "content": "test snippet", "source": "test"}]
        display_results("Test message", data=test_data)
        captured = capsys.readouterr()

        assert "Chatbot: Test message" in captured.out
        assert "- item1" in captured.out
        assert "- item2" in captured.out
        assert "Snippet from test" in captured.out

    def test_display_results_handles_refinement(self, capsys):
        """Test that display_results shows refinement prompts when needed."""
        from chatbot import display_results

        # Test with refinement needed
        display_results("Test message", needs_refinement=True)
        captured = capsys.readouterr()

        assert "Test message" in captured.out
        assert "provide more details" in captured.out.lower()

    def test_display_results_no_refinement(self, capsys):
        """Test that display_results doesn't show refinement when not needed."""
        from chatbot import display_results

        display_results("Test message", needs_refinement=False)
        captured = capsys.readouterr()

        assert "Test message" in captured.out
        assert "provide more details" not in captured.out.lower()