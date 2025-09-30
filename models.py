"""Data models for the ecommerce chatbot."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


# Enums for type safety
class ActionType(Enum):
    """Enumeration of possible chatbot actions."""
    QUERY = "QUERY"
    DISPLAY = "DISPLAY"
    SUMMARIZE = "SUMMARIZE"


class CollectionType(Enum):
    """Enumeration of available ChromaDB collections."""
    PRODUCT_META = "product_meta"
    PRODUCT_REVIEW = "product_review"


# Data classes for structured data
@dataclass
class GeminiResponse:
    """Structured representation of Gemini's YAML response."""
    action: ActionType
    parameters: Dict[str, Any]


@dataclass
class QueryParameters:
    """Parameters for QUERY actions."""
    query_text: str
    collection: CollectionType
    n_results: int = 5


@dataclass
class DisplayParameters:
    """Parameters for DISPLAY actions."""
    message: str
    data: Optional[List[Any]] = None
    snippet_source: Optional[str] = None
    needs_refinement: bool = False


@dataclass
class SummarizeParameters:
    """Parameters for SUMMARIZE actions."""
    text_to_summarize: str


@dataclass
class ChatbotConfig:
    """Configuration class for chatbot settings."""
    max_retries: int = 3
    exit_command: str = "exit"
    default_query_results: int = 5
    comprehensive_meta_results: int = 20
    comprehensive_review_results: int = 30