"""Streamlit chat interface for RAG system."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
import logging

logger = logging.getLogger(__name__)
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.rag.rag_pipeline import RAGPipeline
    from src.utils.config_loader import load_config
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


class ChatInterface:
    """Streamlit chat interface for complaint analysis."""

    def __init__(self):
        self.config = load_config()
        self.rag_pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize RAG pipeline."""
        try:
            self.rag_pipeline = RAGPipeline(self.config)

            # Try to load existing vector store
            # os.path.join("../../", config["vector_store"]["metadata_path"])
            index_path = os.path.join(
                project_root, self.config["vector_store"]["index_path"]
            )
            metadata_path = os.path.join(
                project_root, self.config["vector_store"]["metadata_path"]
            )
            # logger.info(f"Attempting to load vector store from {index_path} ")

            if Path(index_path).exists() and Path(metadata_path).exists():
                self.rag_pipeline.load_vector_store(index_path, metadata_path)
                st.success("Vector store loaded successfully!")
            else:
                st.warning(
                    "Vector store not found. Please build it first using the build script."
                )

        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")

    def render_sidebar(self):
        """Render sidebar with system information."""
        st.sidebar.title("🏦 CrediTrust AI Assistant")
        st.sidebar.markdown("---")

        st.sidebar.subheader("📊 System Status")
        if self.rag_pipeline and self.rag_pipeline.embedding_manager.index:
            st.sidebar.success("RAG System Ready")
            st.sidebar.info(
                f"📚 {len(self.rag_pipeline.embedding_manager.metadata)} chunks indexed"
            )
        else:
            st.sidebar.error("RAG System Not Ready")

        st.sidebar.markdown("---")
        st.sidebar.subheader("--- Sample Questions ---")
        sample_questions = [
            "What are the main credit card issues?",
            "What problems do customers face with loans?",
            "Which product has the most complaints?",
            "What are common billing disputes?",
            "Why are people unhappy with BNPL?",
        ]

        for question in sample_questions:
            if st.sidebar.button(question, key=f"sample_{hash(question)}"):
                st.session_state.current_question = question

    def render_chat_interface(self):
        """Render main chat interface."""
        st.title("🤖 Customer Complaint Analysis Assistant")
        st.markdown(
            "Ask me anything about customer complaints across our financial products!"
        )

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_question" not in st.session_state:
            st.session_state.current_question = ""

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Display sources if available
                if message["role"] == "assistant" and "sources" in message:
                    self._display_sources(message["sources"])

        # Chat input
        question = (
            st.chat_input("Ask about customer complaints...")
            or st.session_state.current_question
        )

        if question:
            # Clear current question
            st.session_state.current_question = ""

            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Generate response
            with st.chat_message("assistant"):
                if self.rag_pipeline and self.rag_pipeline.embedding_manager.index:
                    try:
                        with st.spinner("Analyzing complaints..."):
                            response = self.rag_pipeline.generate_response(question)

                        # Display answer in a beautiful format
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f4f8; border-radius: 10px; padding: 20px; margin-bottom: 10px; border: 1px solid #e0e0e0;">
                                <h4 style="color: #2a5298; margin-top: 0;">💡 Assistant's Answer</h4>
                                <div style="font-size: 1.1em; color: #222;">
                                    {response["answer"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Display sources
                        if response["sources"]:
                            self._display_sources(response["sources"])

                        # Add to session state
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": response["sources"],
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Error generating response for '{question}': {e}",
                            exc_info=True,
                        )
                        error_msg = "I encountered an error while processing your question. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                else:
                    error_msg = "RAG system is not ready. Please ensure the vector store is built."
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    def _display_sources(self, sources):
        """Display source information."""
        if not sources:
            return

        with st.expander(
            f"📚 View Sources ({len(sources)} complaints)", expanded=False
        ):
            for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                st.markdown(
                    f"**Source {i}** (Similarity: {source['similarity_score']:.2f})"
                )
                st.markdown(f"- **Product**: {source['product']}")
                st.markdown(f"- **Issue**: {source['issue']}")
                st.markdown(f"- **Date**: {source['date']}")
                st.markdown(f"- **Text**: {source['chunk_text'][:300]}...")
                st.markdown("---")

    def render_clear_button(self):
        """Render clear conversation button."""
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(
            page_title="Complaint AI Assistant", page_icon="🏦", layout="wide"
        )

        # Custom CSS
        st.markdown(
            """
        <style>
        .stChatMessage {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        .stExpander {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Layout
        col1, col2 = st.columns([3, 1])

        with col1:
            self.render_chat_interface()
            self.render_clear_button()

        with col2:
            self.render_sidebar()


def main():
    """Main function to run the chat interface."""
    chat_app = ChatInterface()
    chat_app.run()


if __name__ == "__main__":
    main()
