"""RAG (Retrieval-Augmented Generation) pipeline implementation.

This module implements a complete RAG pipeline for analyzing customer complaints
in the financial services domain. It combines document retrieval with language
model generation to provide contextual answers to user queries.

The pipeline includes:
- Document chunking and embedding
- Semantic search for relevant context
- LLM-based response generation
- Streaming response capabilities
"""

from typing import List, Dict, Any
import sys
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.embeddings import EmbeddingManager
from src.rag.text_processor import TextProcessor


class RAGPipeline:
    """Complete RAG pipeline for financial complaint analysis.
    
    This class orchestrates the entire RAG workflow:
    1. Text processing and chunking
    2. Vector embedding and storage
    3. Semantic search and retrieval
    4. Language model response generation
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters
        embedding_manager (EmbeddingManager): Handles vector operations
        text_processor (TextProcessor): Processes and chunks text
        llm (InferenceClient): Language model for response generation
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG pipeline with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model settings (embedding model, chunk size, etc.)
                - vector store paths
                - retrieval parameters
        """
        self.config = config
        
        # Initialize embedding manager for vector operations
        self.embedding_manager = EmbeddingManager(config["model"]["embedding_model"])
        
        # Initialize text processor for chunking documents
        self.text_processor = TextProcessor(
            chunk_size=config["model"]["chunk_size"],
            chunk_overlap=config["model"]["chunk_overlap"],
        )
        
        # Initialize language model for response generation
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model client.
        
        Sets up the Hugging Face Inference Client for generating responses.
        Falls back gracefully if initialization fails.
        """
        try:
            # Initialize Hugging Face Inference Client with DeepSeek model
            self.llm = InferenceClient(
                model="deepseek-ai/DeepSeek-V3-0324",  # High-performance language model
                token=os.getenv("HF_TOKEN")  # API token from environment
            )
            print("Hugging Face model initialized successfully")
        except Exception as e:
            # Graceful fallback - system will use fallback responses
            self.llm = None
            print(f"Model initialization failed: {e}. Using fallback responses.")

    def build_vector_store(self, df) -> None:
        """Build vector store from complaint dataframe.
        
        Processes the dataframe into text chunks and creates a searchable
        vector index for semantic retrieval.
        
        Args:
            df (pandas.DataFrame): Dataframe containing complaint data with
                                 'cleaned_narrative' column
        """
        # Process dataframe into chunks with metadata
        chunks = self.text_processor.process_dataframe(df, "cleaned_narrative")
        
        # Build FAISS vector index from chunks
        self.embedding_manager.build_vector_store(chunks)

    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant complaint contexts for a user query.
        
        Uses semantic search to find the most relevant complaint chunks
        that can provide context for answering the query.
        
        Args:
            query (str): User's question or query
            top_k (int, optional): Number of contexts to retrieve.
                                 Defaults to config value.
        
        Returns:
            List[Dict[str, Any]]: List of relevant complaint chunks with metadata
        """
        # Use configured top_k if not specified
        top_k = top_k or self.config["model"]["top_k_retrieval"]
        
        # Perform semantic search in vector store
        return self.embedding_manager.search(query, top_k)

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format retrieved contexts into a readable prompt string.
        
        Converts the list of complaint contexts into a formatted string
        suitable for inclusion in the language model prompt.
        
        Args:
            context (List[Dict[str, Any]]): Retrieved complaint contexts
        
        Returns:
            str: Formatted context string for LLM prompt
        """
        return "\n\n".join(
            f"Complaint {i+1} ({ctx['product']}): {ctx['chunk_text'][:300]}..."
            for i, ctx in enumerate(context)
        )

    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a complete response using the RAG pipeline.
        
        Orchestrates the full RAG process: retrieval, context formatting,
        and response generation.
        
        Args:
            query (str): User's question about financial complaints
        
        Returns:
            Dict[str, Any]: Response containing:
                - answer (str): Generated response text
                - sources (List[Dict]): Source complaint contexts
                - confidence (float): Confidence score (0-1)
        """
        # Step 1: Retrieve relevant contexts
        context = self.retrieve_context(query)
        
        # Handle case with no relevant context found
        if not context:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "confidence": 0.0,
            }

        # Step 2: Prepare messages for language model
        messages = [{
            "role": "system",
            "content": f"You are a financial analyst. Use this context: {self._format_context(context)}"
        }, {
            "role": "user",
            "content": query
        }]

        # Step 3: Generate response using LLM
        answer = self._generate_llm_response(messages, query, context)
        
        # Step 4: Calculate confidence based on top similarity score
        confidence = min(context[0].get("similarity_score", 0.0), 1.0) if context else 0.0

        return {"answer": answer, "sources": context, "confidence": confidence}
    
    def generate_streaming_response(self, query: str):
        """Generate a streaming response using the RAG pipeline.
        
        Provides real-time response generation for better user experience.
        Yields response chunks as they are generated.
        
        Args:
            query (str): User's question about financial complaints
        
        Yields:
            Dict[str, Any]: Response chunks containing:
                - answer (str): Partial response text
                - sources (List[Dict]): Source contexts (on first chunk)
                - confidence (float): Confidence score
                - done (bool): Whether streaming is complete
        """
        # Retrieve relevant contexts for the query
        context = self.retrieve_context(query)
        
        # Handle no context case
        if not context:
            yield {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "confidence": 0.0,
                "done": True
            }
            return

        # Prepare messages for streaming generation
        messages = [{
            "role": "system",
            "content": f"You are a financial analyst. Use this context: {self._format_context(context)}"
        }, {
            "role": "user",
            "content": query
        }]

        # Calculate confidence from top similarity score
        confidence = min(context[0].get("similarity_score", 0.0), 1.0) if context else 0.0
        
        # Stream response chunks as they are generated
        for chunk in self._generate_streaming_llm_response(messages, query, context):
            yield {
                "answer": chunk,
                "sources": context,
                "confidence": confidence,
                "done": False
            }
        
        # Signal completion
        yield {"done": True}

    def _generate_llm_response(self, messages: List[Dict], query: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using language model or fallback.
        
        Attempts to generate a response using the LLM, with graceful
        fallback to a context-based response if the LLM fails.
        
        Args:
            messages (List[Dict]): Chat messages for the LLM
            query (str): Original user query
            context (List[Dict[str, Any]]): Retrieved contexts
        
        Returns:
            str: Generated response text
        """
        # Use fallback if LLM not available
        if not self.llm:
            return self._fallback_response(query, context)
            
        try:
            # Generate response using Hugging Face model
            response = self.llm.chat_completion(
                messages=messages,
                # temperature=0.7  # Balance creativity and consistency
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM generation error: {e}")
            # Fallback to context-based response
            return self._fallback_response(query, context)
    
    def _generate_streaming_llm_response(self, messages: List[Dict], query: str, context: List[Dict[str, Any]]):
        """Generate streaming response using language model.
        
        Streams response tokens in real-time for better user experience.
        Includes multiple fallback mechanisms for robustness.
        
        Args:
            messages (List[Dict]): Chat messages for the LLM
            query (str): Original user query
            context (List[Dict[str, Any]]): Retrieved contexts
        
        Yields:
            str: Response text chunks as they are generated
        """
        # Use fallback if LLM not available
        if not self.llm:
            yield self._fallback_response(query, context)
            return
            
        try:
            # Attempt streaming generation
            stream = self.llm.chat_completion(
                messages=messages,
                stream=True  # Enable streaming mode
            )
            
            # Process each chunk from the stream
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Streaming error: {e}")
            # Final fallback to context-based response
            yield self._fallback_response(query, context)

    def _fallback_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate fallback response when LLM is unavailable.
        
        Creates a basic response using the retrieved context when the
        language model cannot generate a response.
        
        Args:
            query (str): Original user query
            context (List[Dict[str, Any]]): Retrieved complaint contexts
        
        Returns:
            str: Fallback response based on available context
        """
        # Handle case with no context
        if not context:
            return "No relevant complaints found for your query."

        # Extract excerpt from top result
        excerpt = context[0].get("chunk_text", "")[:150]
        
        # Generate basic response with context summary
        if excerpt:
            return f"Based on {len(context)} relevant complaints: {excerpt}..."
        else:
            return f"Found {len(context)} relevant complaints related to your query."

    def save_vector_store(self, index_path: str, metadata_path: str) -> None:
        """Save the vector store to disk.
        
        Persists the FAISS index and metadata for later use.
        
        Args:
            index_path (str): Path to save the FAISS index
            metadata_path (str): Path to save the metadata
        """
        self.embedding_manager.save_vector_store(index_path, metadata_path)

    def load_vector_store(self, index_path: str, metadata_path: str) -> None:
        """Load the vector store from disk.
        
        Loads a previously saved FAISS index and metadata.
        
        Args:
            index_path (str): Path to the FAISS index file
            metadata_path (str): Path to the metadata file
        """
        self.embedding_manager.load_vector_store(index_path, metadata_path)