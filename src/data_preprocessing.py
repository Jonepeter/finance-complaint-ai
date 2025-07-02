"""
Data cleaning and preprocessing functions for complaint data.
"""

# Functions will be implemented here. 
import re

def clean_narrative(text):
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
        # Lowercase
        text = text.lower()
        # Remove boilerplate phrases (expand as needed)
        boilerplate_patterns = [
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
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text)
        # Remove special characters except basic punctuation
        text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    except Exception as e:
        print(f"Error cleaning narrative: {e}")
