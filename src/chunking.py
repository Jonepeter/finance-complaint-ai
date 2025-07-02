"""
Text chunking logic for splitting complaint narratives.
"""

# Functions will be implemented here. 
"""
Text chunking logic for splitting complaint narratives.
"""

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits a long text into overlapping chunks for better embedding performance.

    Args:
        text (str): The input narrative text.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
        List[str]: List of text chunks.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == text_length:
            break
        start += chunk_size - chunk_overlap

    return chunks
