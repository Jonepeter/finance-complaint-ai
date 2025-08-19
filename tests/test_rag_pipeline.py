import unittest
from unittest.mock import Mock, patch
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k_retrieval": 3
            }
        }
        
    @patch('src.rag.rag_pipeline.EmbeddingManager')
    @patch('src.rag.rag_pipeline.TextProcessor')
    def test_init(self, mock_text_processor, mock_embedding_manager):
        pipeline = RAGPipeline(self.config)
        self.assertIsNotNone(pipeline.config)
        mock_embedding_manager.assert_called_once()
        mock_text_processor.assert_called_once()

    @patch('src.rag.rag_pipeline.EmbeddingManager')
    @patch('src.rag.rag_pipeline.TextProcessor')
    def test_retrieve_context(self, mock_text_processor, mock_embedding_manager):
        pipeline = RAGPipeline(self.config)
        mock_embedding_manager.return_value.search.return_value = [
            {"chunk_text": "test", "similarity_score": 0.8}
        ]
        
        result = pipeline.retrieve_context("test query")
        self.assertIsInstance(result, list)

    @patch('src.rag.rag_pipeline.EmbeddingManager')
    @patch('src.rag.rag_pipeline.TextProcessor')
    def test_format_context(self, mock_text_processor, mock_embedding_manager):
        pipeline = RAGPipeline(self.config)
        context = [
            {"product": "Credit Card", "chunk_text": "Test complaint text"},
            {"product": "Loan", "chunk_text": "Another complaint"}
        ]
        
        result = pipeline._format_context(context)
        self.assertIn("Credit Card", result)
        self.assertIn("Loan", result)

    @patch('src.rag.rag_pipeline.EmbeddingManager')
    @patch('src.rag.rag_pipeline.TextProcessor')
    def test_fallback_response(self, mock_text_processor, mock_embedding_manager):
        pipeline = RAGPipeline(self.config)
        context = [{"chunk_text": "Test complaint text"}]
        
        result = pipeline._fallback_response("test query", context)
        self.assertIn("Based on", result)


if __name__ == '__main__':
    unittest.main()