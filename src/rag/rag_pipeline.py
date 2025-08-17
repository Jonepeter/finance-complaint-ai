"""RAG pipeline implementation."""

from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.rag.embeddings import EmbeddingManager
from src.rag.text_processor import TextProcessor


class RAGPipeline:
    """Complete RAG pipeline for complaint analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_manager = EmbeddingManager(config["model"]["embedding_model"])
        self.text_processor = TextProcessor(
            chunk_size=config["model"]["chunk_size"],
            chunk_overlap=config["model"]["chunk_overlap"],
        )

        # Initialize LLM
        self.llm = None
        self.tokenizer = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            import openai
            from dotenv import load_dotenv
            
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            
            if api_key:
                openai.api_key = api_key
                self.use_openai = True
                print("OpenAI client initialized for LLM responses")
            else:
                self.use_openai = False
                print("No OpenAI API key found. Using fallback responses.")
        except ImportError:
            self.use_openai = False
            print("Using fallback text generation without external LLM")
        self.tokenizer = None
        self.llm = None

    def build_vector_store(self, df) -> None:
        """Build vector store from dataframe."""
        chunks = self.text_processor.process_dataframe(df, "cleaned_narrative")
        self.embedding_manager.build_vector_store(chunks)

    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for query."""
        if top_k is None:
            top_k = self.config["model"]["top_k_retrieval"]

        return self.embedding_manager.search(query, top_k)

    def create_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM."""
        context_text = "\\n\\n".join(
            [
                f"Complaint {i+1} ({ctx['product']}): {ctx['chunk_text'][:300]}..."
                for i, ctx in enumerate(context)
            ]
        )

        prompt = f"""You are a financial analyst assistant. 
                     Your task is to analyze customer complaints 
                     and provide insights.   
                     
                     Based on the following complaint excerpts, 
                     answer the user's question. 
                     Be concise and focus on key insights.

                    Context:
                    {context_text}

                    Question: {query}

                    Answer: """

        return prompt

    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response using the RAG pipeline.
        
        Raises:
            Exception: If an error occurs during processing.
        """
        # Retrieve relevant context
        context = self.retrieve_context(query)

        if not context:
            return {
                "answer": "I'm sorry, I don't have enough information to answer this question based on the available complaints.",
                "sources": [],
                "confidence": 0.0,
            }

        # Create prompt for LLM
        prompt = self.create_prompt(query, context)

        # Generate LLM response
        answer = self._generate_llm_response(prompt, query, context)

        # Confidence is based on the top context similarity score, capped at 1.0
        confidence = (
            min(context[0].get("similarity_score", 0.0), 1.0) if context else 0.0
        )

        return {"answer": answer, "sources": context, "confidence": confidence}

    def _generate_llm_response(
        self, prompt: str, query: str, context: List[Dict[str, Any]]
    ) -> str:
        """Generate response using LLM or fallback."""
        if hasattr(self, "use_openai") and self.use_openai:
            try:
                import openai

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700,
                    temperature=0.9,
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API error: {e}")

        return self._fallback_response(query, context)

    def _fallback_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate fallback response when LLM unavailable."""
        if not context:
            return "No relevant complaints found."

        products = list(
            {ctx.get("product", "") for ctx in context if ctx.get("product")}
        )
        issues = list({ctx.get("issue", "") for ctx in context if ctx.get("issue")})

        response = f"Based on {len(context)} complaints"
        
        excerpt = context[0].get("chunk_text", "")[:150]
        if excerpt:
            response += f':- "{excerpt}..."'

        return response

    def save_vector_store(self, index_path: str, metadata_path: str) -> None:
        """Save the vector store to disk."""
        self.embedding_manager.save_vector_store(index_path, metadata_path)

    def load_vector_store(self, index_path: str, metadata_path: str) -> None:
        """Load the vector store from disk."""
        self.embedding_manager.load_vector_store(index_path, metadata_path)
