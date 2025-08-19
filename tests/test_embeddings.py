import unittest
from unittest.mock import Mock, patch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.embeddings import EmbeddingManager


class TestEmbeddingManager(unittest.TestCase):
    def setUp(self):
        self.model_name = "all-MiniLM-L6-v2"
        
    @patch('sentence_transformers.SentenceTransformer')
    def test_init(self, mock_sentence_transformer):
        manager = EmbeddingManager(self.model_name)
        self.assertEqual(manager.model_name, self.model_name)
        mock_sentence_transformer.assert_called_once_with(self.model_name)
        
    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_texts(self, mock_sentence_transformer):
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model
        
        manager = EmbeddingManager(self.model_name)
        result = manager.encode_texts(["test text"])
        
        self.assertIsInstance(result, np.ndarray)
        mock_model.encode.assert_called_once()
        
    @patch('sentence_transformers.SentenceTransformer')
    @patch('faiss.IndexFlatIP')
    def test_build_vector_store(self, mock_faiss_index, mock_sentence_transformer):
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model
        
        manager = EmbeddingManager(self.model_name)
        chunks = [{"chunk_text": "test", "product": "Credit Card"}]
        
        manager.build_vector_store(chunks)
        self.assertIsNotNone(manager.index)
        self.assertEqual(len(manager.metadata), 1)


if __name__ == '__main__':
    unittest.main()