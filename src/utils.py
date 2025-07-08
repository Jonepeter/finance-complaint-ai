"""
Utility functions for the finance-complaint-ai project.
Add general-purpose helpers here.
"""

# Add utility functions below as needed 
def load_data(filepath):
    """
    Load data from a file. Supports CSV, JSON, Parquet, and Excel formats.

    Args:
        filepath (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """
    import os
    import pandas as pd

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext == ".json":
        return pd.read_json(filepath)
    elif ext == ".parquet":
        return pd.read_parquet(filepath)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
