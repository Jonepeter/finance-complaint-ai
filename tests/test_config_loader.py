import unittest
from unittest.mock import patch, mock_open
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config


class TestConfigLoader(unittest.TestCase):
    
    @patch('builtins.open', new_callable=mock_open, read_data="""
model:
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 500
  chunk_overlap: 50
  top_k_retrieval: 3

vector_store:
  index_path: "vector_store/index.faiss"
  metadata_path: "vector_store/metadata.pkl"
""")
    @patch('os.path.exists')
    def test_load_config_success(self, mock_exists, mock_file):
        mock_exists.return_value = True
        
        config = load_config()
        
        self.assertIn('model', config)
        self.assertIn('vector_store', config)
        self.assertEqual(config['model']['embedding_model'], 'all-MiniLM-L6-v2')
        self.assertEqual(config['model']['chunk_size'], 500)
        
    @patch('os.path.exists')
    def test_load_config_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            load_config()


if __name__ == '__main__':
    unittest.main()