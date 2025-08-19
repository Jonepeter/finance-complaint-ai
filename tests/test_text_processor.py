import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        
    def test_init(self):
        self.assertEqual(self.processor.chunk_size, 100)
        self.assertEqual(self.processor.chunk_overlap, 20)
        
    def test_clean_text(self):
        dirty_text = "  This is a TEST text with   extra spaces!  "
        clean_text = self.processor.clean_text(dirty_text)
        self.assertEqual(clean_text, "this is a test text with extra spaces!")
        
    def test_chunk_text(self):
        text = "This is a long text that should be chunked into smaller pieces for testing purposes."
        chunks = self.processor.chunk_text(text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
    def test_process_dataframe(self):
        df = pd.DataFrame({
            'cleaned_narrative': ['First complaint text', 'Second complaint text'],
            'product': ['Credit Card', 'Loan'],
            'issue': ['Billing', 'Payment']
        })
        
        chunks = self.processor.process_dataframe(df, 'cleaned_narrative')
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIn('chunk_text', chunks[0])
        self.assertIn('product', chunks[0])


if __name__ == '__main__':
    unittest.main()