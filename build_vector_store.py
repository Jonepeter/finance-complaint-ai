"""Build vector store from complaint data."""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append("src")

from rag.rag_pipeline import RAGPipeline
from utils.config_loader import load_config


def main():
    """Build vector store from complaint data."""
    print("Building CrediTrust Complaint Vector Store...")

    # Load configuration
    config = load_config()

    # Load data - use only working CSV file
    data_paths = [Path("data/raw/complaints.csv")]

    df = None
    for data_path in data_paths:
        if data_path.exists():
            print(f"Loading complaint data from {data_path}...")
            try:
                if data_path.suffix == ".csv":
                    # Try different encodings
                    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(
                                data_path, encoding=encoding, low_memory=False
                            )
                            print(f"Successfully loaded with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If all encodings fail, use utf-8 with error handling
                        df = pd.read_csv(
                            data_path,
                            encoding="utf-8",
                            errors="ignore",
                            low_memory=False,
                        )
                else:
                    df = pd.read_parquet(data_path)
                print(f"Successfully loaded {len(df)} records")
                break
            except Exception as e:
                print(f"Failed to load {data_path}: {e}")
                continue

    if df is None:
        print("No valid data file found. Please ensure complaint data exists.")
        return

    # Filter for target products
    target_products = config["products"]["target_products"]
    df_filtered = df[df["Product"].isin(target_products)].copy()

    # Clean narratives
    narrative_col = "Consumer complaint narrative"
    if narrative_col not in df_filtered.columns:
        print("Consumer complaint narrative column not found.")
        return

    # Remove empty narratives
    df_filtered = df_filtered[df_filtered[narrative_col].notna()].copy()
    df_filtered["cleaned_narrative"] = df_filtered[narrative_col].fillna("").astype(str)

    print(f"Processing {len(df_filtered)} complaints...")

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(config)

    # Build vector store
    print("Building vector store...")
    rag_pipeline.build_vector_store(df_filtered)

    # Save vector store
    index_path = config["vector_store"]["index_path"]
    metadata_path = config["vector_store"]["metadata_path"]

    print("Saving vector store...")
    rag_pipeline.save_vector_store(index_path, metadata_path)

    print("Vector store built successfully!")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
