import pytest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def chroma_db_setup():
    """Fixture to ensure ChromaDB is available for testing."""
    try:
        from chroma_db_config import get_chromadb
        client, product_meta_collection, product_review_collection = get_chromadb()
        return client, product_meta_collection, product_review_collection
    except Exception as e:
        pytest.skip(f"ChromaDB setup failed: {e}")
        return None, None, None