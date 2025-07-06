"""
Data cleaning and preprocessing class for complaint data in finance-complaint-ai.
Provides methods for loading and cleaning narrative data.
"""

# Functions will be implemented here. 
import re
import pandas as pd

class ComplaintDataPreprocessor:
    """
    Class for cleaning and loading consumer complaint narrative data.
    """
    def __init__(self):
        # Define boilerplate patterns once for reuse
        self.boilerplate_patterns = [
            r"^i am writing to (file|submit|lodge) a complaint[., ]*",
            r"^to whom it may concern[., ]*",
            r"^dear (sir|madam)[., ]*",
            r"^hello[., ]*",
            r"^hi[., ]*",
            r"^greetings[., ]*",
            r"^this is regarding[., ]*",
            r"^i would like to (report|complain about)[., ]*",
            r"^my name is [a-z ]+[., ]*",
            r"^i am contacting you[., ]*",
            r"^i am reaching out[., ]*",
        ]

    def clean_narrative(self, text):
        """
        Cleans a consumer complaint narrative for improved text processing and embedding quality.
        Steps performed:
            - Converts text to lowercase
            - Removes common boilerplate phrases (e.g., greetings, introductions)
            - Removes special characters except basic punctuation (.,!?'-)
            - Collapses multiple spaces into one
            - Strips leading/trailing whitespace
        Args:
            text (str): The input narrative text.
        Returns:
            str: The cleaned narrative text. Returns an empty string if input is not a string or on error.
        """
        try:
            if not isinstance(text, str):
                return ""
            text = text.lower()
            for pattern in self.boilerplate_patterns:
                text = re.sub(pattern, "", text)
            text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            return text
        except Exception as e:
            print(f"Error cleaning narrative: {e}")
            return ""

    def load_data(self, filepath, text_column=None):
        """
        Loads complaint data from a CSV file and returns a list of narrative texts using pandas.

        Args:
            filepath (str): Path to the CSV file.
            text_column (str): Name of the column containing the narrative text.

        Returns:
            List[str]: List of narrative texts. Returns an empty list if file cannot be read or column is missing.
        """
        try:
            df = pd.read_csv(filepath)
            if text_column not in df.columns:
                print(f"Column '{text_column}' not found in {filepath}.")
                return []
            narratives = df[text_column].dropna().astype(str)
            narratives = [text for text in narratives if text.strip()]
            return narratives
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return []

