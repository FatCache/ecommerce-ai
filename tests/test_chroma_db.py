import pytest
from unittest.mock import patch, MagicMock


class TestChromaDBConfiguration:
    """Test suite for ChromaDB configuration and connectivity."""

    def test_get_chromadb_returns_three_values(self, chroma_db_setup):
        """Test that get_chromadb returns client, meta_collection, and review_collection."""
        client, meta_collection, review_collection = chroma_db_setup

        assert client is not None, "ChromaDB client should not be None"
        assert meta_collection is not None, "Product meta collection should not be None"
        assert review_collection is not None, "Product review collection should not be None"

    def test_collections_have_correct_names(self, chroma_db_setup):
        """Test that collections have the expected names."""
        client, meta_collection, review_collection = chroma_db_setup

        assert meta_collection.name == "product_meta", f"Expected 'product_meta', got '{meta_collection.name}'"
        assert review_collection.name == "product_review", f"Expected 'product_review', got '{review_collection.name}'"

    def test_collections_have_metadata(self, chroma_db_setup):
        """Test that collections have proper metadata descriptions."""
        client, meta_collection, review_collection = chroma_db_setup

        meta_metadata = meta_collection.metadata or {}
        review_metadata = review_collection.metadata or {}

        assert "description" in meta_metadata, "Product meta collection should have description metadata"
        assert "description" in review_metadata, "Product review collection should have description metadata"

        assert "Product metadata collection" in meta_metadata["description"]
        assert "Product review collection" in review_metadata["description"]

    def test_collections_support_basic_operations(self, chroma_db_setup):
        """Test that collections support basic ChromaDB operations."""
        client, meta_collection, review_collection = chroma_db_setup

        # Test count operation (should not raise exception)
        try:
            meta_count = meta_collection.count()
            review_count = review_collection.count()
            assert isinstance(meta_count, int), "Count should return an integer"
            assert isinstance(review_count, int), "Count should return an integer"
        except Exception as e:
            pytest.fail(f"Collection count operation failed: {e}")

    def test_query_operation_structure(self, chroma_db_setup):
        """Test that query operations return properly structured results."""
        client, meta_collection, review_collection = chroma_db_setup

        # Test with empty query - should return empty results but proper structure
        try:
            results = meta_collection.query(query_texts=[""], n_results=1)
            assert "ids" in results, "Query results should contain 'ids' key"
            assert "documents" in results, "Query results should contain 'documents' key"
            assert "metadatas" in results, "Query results should contain 'metadatas' key"
            assert isinstance(results["ids"], list), "'ids' should be a list"
            assert isinstance(results["documents"], list), "'documents' should be a list"
            assert isinstance(results["metadatas"], list), "'metadatas' should be a list"
        except Exception as e:
            pytest.fail(f"Query operation failed: {e}")

    @patch('chromadb.PersistentClient')
    def test_get_chromadb_handles_missing_directory(self, mock_client):
        """Test that get_chromadb handles missing database directory gracefully."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock collections
        mock_meta_collection = MagicMock()
        mock_meta_collection.name = "product_meta"
        mock_review_collection = MagicMock()
        mock_review_collection.name = "product_review"

        mock_client_instance.get_or_create_collection.side_effect = [mock_meta_collection, mock_review_collection]

        from chroma_db_config import get_chromadb

        client, meta_col, review_col = get_chromadb()

        mock_client.assert_called_once_with(path="./chromadbs/chromadb_v1")
        assert client == mock_client_instance
        assert meta_col == mock_meta_collection
        assert review_col == mock_review_collection

    def test_collections_are_different_objects(self, chroma_db_setup):
        """Test that meta and review collections are separate objects."""
        client, meta_collection, review_collection = chroma_db_setup

        assert meta_collection is not review_collection, "Collections should be different objects"
        assert meta_collection.name != review_collection.name, "Collections should have different names"


class TestChromaDBIntegration:
    """Integration tests that require actual database connectivity."""

    @pytest.mark.integration
    def test_real_database_operations(self, chroma_db_setup):
        """Test actual database operations (marked as integration test)."""
        client, meta_collection, review_collection = chroma_db_setup

        # Test basic connectivity by attempting to get collection info
        try:
            meta_info = meta_collection.count()
            review_info = review_collection.count()
            # If we get here without exception, the database is accessible
            assert True
        except Exception as e:
            pytest.fail(f"Database connectivity test failed: {e}")

    @pytest.mark.integration
    def test_query_returns_reasonable_results(self, chroma_db_setup):
        """Test that queries return reasonable result structures."""
        client, meta_collection, review_collection = chroma_db_setup

        try:
            # Query for a common term
            results = meta_collection.query(query_texts=["test"], n_results=5)

            # Verify result structure
            assert len(results["ids"]) <= 5, "Should return at most 5 results"
            assert len(results["documents"]) <= 5, "Documents list should match ids length"
            assert len(results["metadatas"]) <= 5, "Metadatas list should match ids length"

            # If there are results, they should have proper structure
            if results["ids"]:
                assert all(isinstance(doc_id, str) for doc_id in results["ids"][0]), "IDs should be strings"

        except Exception as e:
            pytest.fail(f"Query structure test failed: {e}")