"""Optimized embedding generation using sentence-transformers all-MiniLM-L6-v2.

This module provides the EmbeddingManager class for generating, storing, searching,
and persisting vector embeddings using the all-MiniLM-L6-v2 model and FAISS.
"""

import os
import logging

# Suppress TensorFlow and ONEDNN warnings for cleaner logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import torch


class EmbeddingManager:
    """
    EmbeddingManager handles embedding generation, vector store creation, search, and persistence
    using the sentence-transformers all-MiniLM-L6-v2 model and FAISS.

    Attributes:
        model_name (str): Name of the embedding model.
        model (SentenceTransformer): Loaded sentence-transformers model.
        dimension (int): Embedding vector dimension.
        index (faiss.Index or None): FAISS index for vector search.
        metadata (list): List of metadata dictionaries for each chunk.
        device (torch.device): Device (cuda or cpu) for model inference.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingManager.

        Args:
            model_name (str): Name of the sentence-transformers model to use.
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = None
            self.metadata = []
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize embedding model '{model_name}': {e}"
            )

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            np.ndarray: Array of embeddings (float32).
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_tensor=False,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
                device=self.device,
            )
            return np.array(embeddings).astype("float32")
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}")

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create a FAISS index for the given embeddings.

        Args:
            embeddings (np.ndarray): Embedding vectors.

        Returns:
            faiss.Index: FAISS index for similarity search.
        """
        try:
            index = faiss.IndexFlatIP(self.dimension)
            index.add(embeddings)
            return index
        except Exception as e:
            raise RuntimeError(f"Error creating FAISS index: {e}")

    def build_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build the vector store (FAISS index and metadata) from text chunks.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries with "chunk_text".
        """
        try:
            texts = [chunk["chunk_text"] for chunk in chunks]
            self.metadata = chunks
            embeddings = self.generate_embeddings(texts)
            self.index = self.create_faiss_index(embeddings)
            print(
                f"Vector store built with {len(texts)} embeddings using {self.model_name}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build vector store: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most similar chunks to the query.

        Args:
            query (str): Query string.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: List of metadata dicts with similarity scores.
        """
        if self.index is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")

        try:
            query_embedding = self.generate_embeddings([query])
            scores, indices = self.index.search(query_embedding, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata) and idx != -1:
                    result = self.metadata[idx].copy()
                    result["similarity_score"] = float(score)
                    results.append(result)
            return results
        except Exception as e:
            raise RuntimeError(f"Error during search: {e}")

    def save_vector_store(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            index_path (str): Path to save the FAISS index.
            metadata_path (str): Path to save the metadata JSON.
        """
        if self.index is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")

        try:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, index_path)
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2, default=str)
            print(f"Vector store saved using {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to save vector store: {e}")

    def load_vector_store(self, index_path: str, metadata_path: str) -> None:
        """
        Load the FAISS index and metadata from disk.

        Args:
            index_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata JSON file.
        """
        if not Path(index_path).exists() or not Path(metadata_path).exists():
            raise FileNotFoundError("Vector store files not found")

        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            print(f"Vector store loaded using {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load vector store: {e}")
