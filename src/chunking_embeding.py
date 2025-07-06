"""
Text chunking and embedding pipeline for splitting and embedding complaint narratives in finance-complaint-ai.
Provides a class-based pipeline for chunking, embedding, and indexing complaint data.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pickle
import os
import faiss

class ComplaintChunkEmbedPipeline:
    """
    Pipeline for loading, chunking, embedding, and indexing complaint narratives.
    """
    def __init__(self, data_path, output_dir, chunk_size=350, chunk_overlap=40, model_name="all-MiniLM-L6-v2", batch_size=64):
        self.data_path = data_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.batch_size = batch_size

    def load_and_preprocess(self):
        """
        Loads and preprocesses the complaint data from a CSV or Parquet file.
        Returns:
            pd.DataFrame: Loaded DataFrame.
        Raises:
            ValueError: If the file extension is not supported.
        """
        try:
            print("========== Step 1: Load and preprocess ==========")
            if self.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file type: {self.data_path}. Only .csv and .parquet are supported.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def chunk_text(self, df):
        """
        Splits complaint narratives into chunks using RecursiveCharacterTextSplitter.
        Args:
            df (pd.DataFrame): DataFrame with complaint narratives.
        Returns:
            list: List of langchain Document objects.
        """
        try:
            print("========== Step 2: Chunk text ==========")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            documents = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                chunks = text_splitter.create_documents(
                    [row['Consumer complaint narrative clean']],
                    metadatas=[{
                        'product': row['Product'],
                        'complaint_id': row['Complaint ID']
                    }]
                )
                documents.extend(chunks)
            return documents
        except Exception as e:
            print(f"Error chunking text: {e}")
            raise

    def embed_texts(self, texts):
        """
        Embeds a list of texts using SentenceTransformer.
        Args:
            texts (list): List of text strings.
        Returns:
            np.ndarray: Embeddings array.
        """
        try:
            print("========== Step 3: Embed in batch ==========")
            model = SentenceTransformer(self.model_name)
            embeddings = model.encode(texts, batch_size=self.batch_size, show_progress_bar=True, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error embedding texts: {e}")
            raise

    def build_faiss_index(self, embeddings):
        """
        Builds a FAISS index from embeddings.
        Args:
            embeddings (np.ndarray): Embeddings array.
        Returns:
            faiss.Index: FAISS index object.
        """
        try:
            print("========== Step 4: Build native FAISS index ==========")
            embedding_dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(embeddings.astype("float32"))
            print(f"FAISS index contains {index.ntotal} vectors.")
            return index
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            raise

    def save_faiss_index_and_metadata(self, index, texts, metadatas):
        """
        Saves the FAISS index and metadata to disk.
        Args:
            index (faiss.Index): FAISS index object.
            texts (list): List of text chunks.
            metadatas (list): List of metadata dicts.
        """
        try:
            print("========== Step 5: Save FAISS index and metadata ==========")
            os.makedirs(self.output_dir, exist_ok=True)
            print("Save index")
            faiss.write_index(index, os.path.join(self.output_dir, "index.faiss"))
            print("Save metadata (for lookup after search)")
            with open(os.path.join(self.output_dir, "metadata.pkl"), "wb") as f:
                pickle.dump({'texts': texts, 'metadatas': metadatas}, f)
            print(" Complete Saved FAISS index and metadata.")
        except Exception as e:
            print(f"Error saving FAISS index and metadata: {e}")
            raise

    def run(self):
        """
        Orchestrates the chunking, embedding, indexing, and saving process.
        """
        try:
            df = self.load_and_preprocess()
            documents = self.chunk_text(df)
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embed_texts(texts)
            index = self.build_faiss_index(embeddings)
            self.save_faiss_index_and_metadata(index, texts, metadatas)
        except Exception as e:
            print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    data_path = '../data/filtered_data.parquet'  # Change to .csv if needed
    output_dir = "../vector_store/creditrust_faiss_native"
    pipeline = ComplaintChunkEmbedPipeline(data_path, output_dir)
    pipeline.run()