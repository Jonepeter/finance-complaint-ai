"""Text processing utilities for RAG pipeline.

This module provides the TextProcessor class for cleaning, chunking, and processing
text data for use in a Retrieval-Augmented Generation (RAG) pipeline.
"""

import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


class TextProcessor:
    """
    Handles text chunking and preprocessing for RAG.

    Attributes:
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
        text_splitter (RecursiveCharacterTextSplitter): The splitter instance for chunking.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the TextProcessor with chunking parameters.

        Args:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Use RecursiveCharacterTextSplitter for flexible chunking
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
        """
        Clean and normalize input text.

        - Converts text to lowercase.
        - Removes boilerplate phrases.
        - Removes unwanted special characters.
        - Collapses multiple spaces.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned and normalized text.
        """
        try:
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

            # Remove unwanted special characters, keep basic punctuation
            text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", " ", text)
            # Collapse multiple spaces into one
            text = re.sub(r"\s+", " ", text).strip()

            return text
        except Exception as e:
            # Log or handle error as needed; for now, return empty string
            print(f"[TextProcessor.clean_text] Error cleaning text: {e}")
            return ""

    def chunk_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks and attach metadata to each chunk.

        Args:
            text (str): The text to split.
            metadata (Dict[str, Any], optional): Metadata to attach to each chunk.

        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries with metadata.
        """
        try:
            if not text or len(text.strip()) < 10:
                return []

            # Split the text into chunks using the configured splitter
            chunks = self.text_splitter.split_text(text)

            chunked_docs = []
            for i, chunk in enumerate(chunks):
                # Only keep meaningful chunks (length > 10)
                if len(chunk.strip()) > 10:
                    chunk_metadata = {
                        "chunk_id": i,
                        "chunk_text": chunk,
                        "chunk_length": len(chunk),
                        **(metadata or {}),
                    }
                    chunked_docs.append(chunk_metadata)

            return chunked_docs
        except Exception as e:
            # Log or handle error as needed; for now, return empty list
            print(f"[TextProcessor.chunk_text] Error chunking text: {e}")
            return []

    def process_dataframe(
        self, df: pd.DataFrame, text_column: str
    ) -> List[Dict[str, Any]]:
        """
        Process an entire DataFrame, cleaning and chunking the specified text column.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            text_column (str): The name of the column containing text to process.

        Returns:
            List[Dict[str, Any]]: List of all chunk dictionaries from the DataFrame.
        """
        all_chunks = []
        try:
            for idx, row in df.iterrows():
                # Clean the text in the specified column
                try:
                    text = self.clean_text(row[text_column])
                except Exception as e:
                    print(
                        f"[TextProcessor.process_dataframe] Error cleaning row {idx}: {e}"
                    )
                    text = ""

                # Gather relevant metadata for each row
                metadata = {
                    "original_index": idx,
                    "product": row.get("Product", ""),
                    "issue": row.get("Issue", ""),
                    "date": row.get("Date received", ""),
                    "company": row.get("Company", ""),
                    "state": row.get("State", ""),
                }

                # Chunk the cleaned text and add to the list
                try:
                    chunks = self.chunk_text(text, metadata)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(
                        f"[TextProcessor.process_dataframe] Error chunking row {idx}: {e}"
                    )
                    continue

            return all_chunks
        except Exception as e:
            # Log or handle error as needed; for now, return empty list
            print(f"[TextProcessor.process_dataframe] Error processing DataFrame: {e}")
            return []
