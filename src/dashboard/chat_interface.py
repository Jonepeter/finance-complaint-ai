"""Streamlit chat interface for RAG system (refined for a better chat experience).

This module provides a Streamlit-based chat interface for interacting with a Retrieval-Augmented Generation (RAG) system.
It includes error handling, user experience improvements, and clear code comments for maintainability.
"""

import os

# Suppress TensorFlow and ONEDNN warnings for a cleaner Streamlit log
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
from pathlib import Path
import sys
import logging

# Set up logger for error tracking and debugging
logger = logging.getLogger(__name__)

# Add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Attempt to import required modules, handle import errors gracefully
try:
    from src.rag.rag_pipeline import RAGPipeline
    from src.utils.config_loader import load_config
    from src.utils.speech_handler import SpeechHandler
except ImportError as e:
    # Show error in Streamlit UI and stop execution if imports fail
    st.error(f"Failed to import required modules: {e}")
    st.stop()


class ChatInterface:
    """Streamlit chat interface for complaint analysis."""

    def __init__(self):
        # Load configuration and initialize pipeline and speech handler
        self.config = load_config()
        self.rag_pipeline = None
        self.speech_handler = SpeechHandler()
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize RAG pipeline and load vector store if available.

        Handles errors during initialization and notifies the user.
        """
        try:
            self.rag_pipeline = RAGPipeline(self.config)
            # Construct paths for vector store index and metadata
            index_path = os.path.join(
                project_root, self.config["vector_store"]["index_path"]
            )
            metadata_path = os.path.join(
                project_root, self.config["vector_store"]["metadata_path"]
            )
            # Check if vector store files exist before loading
            if Path(index_path).exists() and Path(metadata_path).exists():
                self.rag_pipeline.load_vector_store(index_path, metadata_path)
                st.toast("Vector store loaded successfully!", icon="✅")
            else:
                st.warning(
                    "Vector store not found. Please build it first using the build script."
                )
        except Exception as e:
            # Log and display error if pipeline initialization fails
            st.error(f"Error initializing RAG pipeline: {e}")

    def render_sidebar(self):
        """Render sidebar with system information and sample questions.

        Provides users with example queries and information about the app.
        """
        # Display sample questions for user inspiration
        st.subheader("💬 Sample Questions")
        sample_questions = [
            "What are the main credit card issues?",
            "What problems do customers face with loans?",
            "Which product has the most complaints?",
            "What are common billing disputes?",
            "Why are people unhappy with BNPL?",
        ]
        for i, question in enumerate(sample_questions):
            # Allow users to click a sample question to populate the input
            if st.write(question, key=f"sample_{i}"):
                st.session_state.current_question = question

        st.divider()

        # About section for app description
        st.subheader("About")
        st.info(
            """
            CrediTrust Financial AI is an interactive chat interface developed by 10 Academy in 2025.
            This application leverages Retrieval-Augmented Generation (RAG) to help users explore and analyze real-world financial complaints.
            Ask questions about credit cards, loans, BNPL, and more to receive instant, AI-powered insights based on real complaint data.
            """
        )

        st.divider()

    def render_chat_interface(self):
        """Render main chat interface with improved UX.

        Handles chat history, user input, speech input, and AI responses.
        """
        # Display header and welcome message
        st.markdown(
            '<div class="header"><h1>💬 CrediTrust Financial AI </h1><p>Start a conversation with our AI assistant</p></div>',
            unsafe_allow_html=True,
        )
        st.info(
            "Welcome to the CrediTrust Financial AI chat! Ask any question about financial complaints, products, or issues and get instant, AI-powered answers."
        )
        # Initialize session state for chat messages and current question
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_question" not in st.session_state:
            st.session_state.current_question = ""

        chat_container = st.container()

        # Display chat history (user and assistant messages)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)
                    # Show sources if present in assistant's message
                    if message["role"] == "assistant" and "sources" in message:
                        self._display_sources(message["sources"])

        # Layout for speech input button and chat input box
        input_col1, input_col2 = st.columns([5, 1])

        # Speech input button (microphone)
        with input_col2:
            if st.button("🎤", help="Click to speak", key="speech_btn"):
                with st.spinner("Listening..."):
                    speech_text = self.speech_handler.listen_once()
                    # Only accept valid speech input (filter out errors)
                    if speech_text and not speech_text.startswith(
                        ("Could not", "Speech service", "No speech", "Error:")
                    ):
                        st.session_state.current_question = speech_text
                        st.rerun()
                    else:
                        st.error(f"Speech failed: {speech_text}")

        # Chat input box for user text input
        with input_col1:
            question = (
                st.chat_input("Type your message here...")
                or st.session_state.current_question
            )

        # If user submits a question (via text or speech)
        if question:
            st.session_state.current_question = ""
            # Add user message to chat history with timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append(
                {"role": "user", "content": question, "timestamp": timestamp}
            )
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)
                    st.markdown(
                        f'<div class="timestamp">{timestamp}</div>',
                        unsafe_allow_html=True,
                    )

            # Generate and display assistant's response
            with chat_container:
                with st.chat_message("assistant"):
                    # Check if RAG pipeline and vector index are ready
                    if self.rag_pipeline and getattr(
                        self.rag_pipeline.embedding_manager, "index", None
                    ):
                        try:
                            full_response = ""
                            sources = []
                            response_placeholder = st.empty()
                            spinner = st.spinner("Thinking...")

                            # Stream response from RAG pipeline for better UX
                            with spinner:
                                for (
                                    chunk
                                ) in self.rag_pipeline.generate_streaming_response(
                                    question
                                ):
                                    if chunk.get("done"):
                                        break
                                    if "answer" in chunk:
                                        full_response += chunk["answer"]
                                        sources = chunk.get("sources", [])
                                        # Live update of assistant's response
                                        response_placeholder.markdown(
                                            f"""
                                                {full_response}
                                            """
                                        )
                            # Display sources if available
                            if sources:
                                self._display_sources(sources)
                            # Add assistant's response to chat history with timestamp
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": full_response,
                                    "sources": sources,
                                    "timestamp": timestamp,
                                }
                            )
                            st.markdown(
                                f'<div class="timestamp">{timestamp}</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            # Log and display error if response generation fails
                            logger.error(
                                f"Error generating response for '{question}': {e}",
                                exc_info=True,
                            )
                            error_msg = "⚠️ I encountered an error while processing your question. Please try again."
                            st.error(error_msg)
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": error_msg,
                                    "timestamp": timestamp,
                                }
                            )
                            st.markdown(
                                f'<div class="timestamp">{timestamp}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        # Handle case where RAG system is not ready (e.g., vector store missing)
                        error_msg = "RAG system is not ready. Please ensure the vector store is built."
                        st.error(error_msg)
                        import datetime

                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_msg,
                                "timestamp": timestamp,
                            }
                        )
                        st.markdown(
                            f'<div class="timestamp">{timestamp}</div>',
                            unsafe_allow_html=True,
                        )

    def _display_sources(self, sources):
        """Display source information in a compact, readable way.

        Shows the top 3 sources (complaints) that contributed to the answer.
        """
        if not sources:
            return
        with st.expander(
            f"📚 View Top Sources ({len(sources)} complaints)", expanded=False
        ):
            for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                st.markdown(
                    f"""
                    <div style="background-color:#f6f8fa; border-radius:7px; padding:10px; margin-bottom:8px; border:1px solid #e0e0e0;">
                        <b>Source {i}</b> <span style="color:#888;">(Similarity: {source.get('similarity_score',0):.2f})</span><br>
                        <b>Product:</b> {source.get('product','-')}<br>
                        <b>Issue:</b> {source.get('issue','-')}<br>
                        <b>Date:</b> {source.get('date','-')}<br>
                        <b>Text:</b> {source.get('chunk_text','')[:300]}...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def run(self):
        """Run the Streamlit app with improved layout and style.

        Sets up the page, applies custom CSS, and arranges the chat and sidebar.
        """
        st.set_page_config(
            page_title="Complaint AI Assistant", page_icon="🤖", layout="wide"
        )

        # Custom CSS for chat bubbles, sidebar, and layout
        st.markdown(
            """
            <style>
            .main {
                padding: 0rem 1rem;
            }
            .stChatMessage {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 0.75rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stChatMessage [data-testid="stChatMessageContent"] {
                padding: 0.5rem 1rem;
            }
            .stChatMessage [data-testid="stChatMessageAvatar"] {
                display: none;
            }
            .user-message {
                background-color: #2b313e;
                color: white;
            }
            .assistant-message {
                background-color: #f0f2f6;
            }
            .chat-input {
                position: fixed;
                bottom: 3rem;
                left: 0;
                right: 0;
                padding: 0 1rem;
                background-color: #0e1117;
                z-index: 999;
            }
            .header {
                background: linear-gradient(90deg, #373B44, #4286f4);
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
                color: white;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .timestamp {
                font-size: 0.7rem;
                color: #9ca3af;
                margin-top: 0.25rem;
            }
            .stExpander {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
            }
            .stButton>button {
                border-radius: 6px;
                font-size: 1em;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Layout: chat on left, sidebar on right
        col1, col2 = st.columns([4, 1], gap="large")

        with col1:
            self.render_chat_interface()

        with col2:
            self.render_sidebar()


def main():
    """Main function to run the chat interface.

    Instantiates the ChatInterface and starts the Streamlit app.
    """
    chat_app = ChatInterface()
    chat_app.run()


if __name__ == "__main__":
    main()
