"""
Text chunking logic for splitting complaint narratives.
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

def load_and_preprocess(data_path):
    """
    Loads and preprocesses the complaint data from a CSV or Parquet file.
    Args:
        data_path (str): Path to the CSV or Parquet file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    Raises:
        ValueError: If the file extension is not supported.
    """
    try:
        print("========== Step 1: Load and preprocess ==========")
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path}. Only .csv and .parquet are supported.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def chunk_text(df, chunk_size=350, chunk_overlap=40):
    """
    Splits complaint narratives into chunks using RecursiveCharacterTextSplitter.
    Args:
        df (pd.DataFrame): DataFrame with complaint narratives.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        list: List of langchain Document objects.
    """
    try:
        print("========== Step 2: Chunk text ==========")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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


def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=64):
    """
    Embeds a list of texts using SentenceTransformer.
    Args:
        texts (list): List of text strings.
        model_name (str): Name of the embedding model.
        batch_size (int): Batch size for embedding.
    Returns:
        np.ndarray: Embeddings array.
    """
    try:
        print("========== Step 3: Embed in batch ==========")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        print(f"Error embedding texts: {e}")
        raise


def build_faiss_index(embeddings):
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


def save_faiss_index_and_metadata(index, texts, metadatas, output_dir):
    """
    Saves the FAISS index and metadata to disk.
    Args:
        index (faiss.Index): FAISS index object.
        texts (list): List of text chunks.
        metadatas (list): List of metadata dicts.
        output_dir (str): Directory to save index and metadata.
    """
    try:
        print("========== Step 5: Save FAISS index and metadata ==========")
        os.makedirs(output_dir, exist_ok=True)
        print("Save index")
        faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
        print("Save metadata (for lookup after search)")
        with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump({'texts': texts, 'metadatas': metadatas}, f)
        print(" Complete Saved FAISS index and metadata.")
    except Exception as e:
        print(f"Error saving FAISS index and metadata: {e}")
        raise


def main():
    """
    Orchestrates the chunking, embedding, indexing, and saving process.
    """
    data_path = '../data/filtered_data.parquet'  # Change to .csv if needed
    output_dir = "../vector_store/creditrust_faiss_native"
    try:
        df = load_and_preprocess(data_path)
        documents = chunk_text(df)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = embed_texts(texts)
        index = build_faiss_index(embeddings)
        save_faiss_index_and_metadata(index, texts, metadatas, output_dir)
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()