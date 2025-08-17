"""Advanced embedding generation and vector store management."""

import warnings
import os
import logging

# Suppress all TensorFlow logging output.
# Use '2' to suppress INFO and WARNING, and '3' to suppress ERROR as well.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Suppress TensorFlow's Python-level warnings by setting the logger's level.
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pickle


class EmbeddingManager:
    """Advanced embedding manager with hybrid search and optimized indexing."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate advanced embeddings with mean pooling and normalization."""
        # Use batch processing with optimal batch size
        batch_size = 32 if len(texts) > 32 else len(texts)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalization
            device=self.device,
        )
        return np.array(embeddings).astype("float32")

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create optimized FAISS index with IVF for large datasets."""
        n_vectors = embeddings.shape[0]

        if n_vectors > 1000:
            # Use IVF index for better performance on large datasets
            nlist = min(int(np.sqrt(n_vectors)), 100)
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(10, nlist)  # Search in 10 clusters
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatIP(self.dimension)
            index.add(embeddings)

        return index

    def build_vector_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Build hybrid vector store with semantic and lexical search."""
        print(f"Processing {len(chunks)} chunks...")

        texts = [chunk["chunk_text"] for chunk in chunks]
        self.metadata = chunks

        print("Generating semantic embeddings...")
        embeddings = self.generate_embeddings(texts)

        print("Building TF-IDF matrix for lexical search...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        print("Creating optimized FAISS index...")
        self.index = self.create_faiss_index(embeddings)

        print(f"Hybrid vector store built with {len(texts)} embeddings")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and lexical similarity."""
        if self.index is None or self.tfidf_matrix is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")

        # Semantic search
        query_embedding = self.generate_embeddings([query])
        semantic_scores, semantic_indices = self.index.search(
            query_embedding, top_k * 2
        )

        # Lexical search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        lexical_indices = np.argsort(lexical_scores)[::-1][: top_k * 2]

        # Combine and rerank results
        combined_results = {}

        # Add semantic results
        for score, idx in zip(semantic_scores[0], semantic_indices[0]):
            if idx < len(self.metadata) and idx != -1:
                combined_results[idx] = {
                    "semantic_score": float(score),
                    "lexical_score": float(lexical_scores[idx]),
                    "metadata": self.metadata[idx],
                }

        # Add lexical results
        for idx in lexical_indices:
            if idx not in combined_results and lexical_scores[idx] > 0.1:
                combined_results[idx] = {
                    "semantic_score": 0.0,
                    "lexical_score": float(lexical_scores[idx]),
                    "metadata": self.metadata[idx],
                }

        # Hybrid scoring (70% semantic, 30% lexical)
        final_results = []
        for idx, data in combined_results.items():
            hybrid_score = 0.7 * data["semantic_score"] + 0.3 * data["lexical_score"]
            result = data["metadata"].copy()
            result["similarity_score"] = hybrid_score
            result["semantic_score"] = data["semantic_score"]
            result["lexical_score"] = data["lexical_score"]
            final_results.append(result)

        # Sort by hybrid score and return top_k
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return final_results[:top_k]

    def save_vector_store(self, index_path: str, metadata_path: str) -> None:
        """Save vector store to disk."""
        if (
            self.index is None
            or self.tfidf_matrix is None
            or self.tfidf_vectorizer is None
        ):
            raise ValueError(
                "Vector store not built or is incomplete. Call build_vector_store first."
            )

        # Create directories
        index_dir = Path(index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # Save TF-IDF vectorizer
        vectorizer_path = index_dir / "tfidf_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)

        print(f"Vector store saved to {index_dir}")

    def load_vector_store(self, index_path: str, metadata_path: str) -> None:
        """Load vector store from disk."""
        index_dir = Path(index_path).parent
        vectorizer_path = index_dir / "tfidf_vectorizer.pkl"

        if not all(
            [Path(index_path).exists(), Path(metadata_path).exists(), vectorizer_path.exists()]
        ):
            raise FileNotFoundError(f"Vector store files not found in {index_dir}")

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load TF-IDF vectorizer and rebuild matrix
        with open(vectorizer_path, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        texts = [chunk["chunk_text"] for chunk in self.metadata]
        self.tfidf_matrix = self.tfidf_vectorizer.transform(texts)

        print(f"Vector store loaded from {index_dir}")
