"""Custom exceptions for the ecommerce chatbot."""


class ChatbotError(Exception):
    """Base exception for chatbot-related errors."""
    pass


class InvalidActionError(ChatbotError):
    """Raised when an invalid action is encountered."""
    pass


class CollectionNotFoundError(ChatbotError):
    """Raised when a requested collection is not found."""
    pass


class GeminiAPIError(ChatbotError):
    """Raised when there's an error with Gemini API calls."""
    pass