"""Text processing utilities for RAG pipeline."""

import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


class TextProcessor:
    """Handles text chunking and preprocessing for RAG."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                ",",
                " ",
                "",
            ],
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or pd.isna(text):
            return ""

        text = str(text).lower()

        # Remove boilerplate patterns
        boilerplate_patterns = [
            r"i am writing to file a complaint",
            r"this is to inform you",
            r"dear sir/madam",
            r"to whom it may concern",
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean special characters
        text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def chunk_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if not text or len(text.strip()) < 10:
            return []

        chunks = self.text_splitter.split_text(text)

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 10:  # Only keep meaningful chunks
                chunk_metadata = {
                    "chunk_id": i,
                    "chunk_text": chunk,
                    "chunk_length": len(chunk),
                    **(metadata or {}),
                }
                chunked_docs.append(chunk_metadata)

        return chunked_docs

    def process_dataframe(
        self, df: pd.DataFrame, text_column: str
    ) -> List[Dict[str, Any]]:
        """Process entire dataframe into chunks."""
        all_chunks = []

        for idx, row in df.iterrows():
            text = self.clean_text(row[text_column])

            metadata = {
                "original_index": idx,
                "product": row.get("Product", ""),
                "issue": row.get("Issue", ""),
                "date": row.get("Date received", ""),
                "company": row.get("Company", ""),
                "state": row.get("State", ""),
            }

            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks
