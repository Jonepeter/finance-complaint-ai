"""Main Streamlit application with multi-page navigation."""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.append("src")

from dashboard.chat_interface import ChatInterface
from dashboard.analytics_dashboard import AnalyticsDashboard


def main():
    """Main application with page navigation."""
    st.set_page_config(
        page_title="CrediTrust AI Platform",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e86ab 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .nav-button {
        width: 100%;
        margin: 0.5rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🏦 CrediTrust AI Platform</h1>
        <p>Intelligent Complaint Analysis & Customer Insights</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")

    page = st.sidebar.selectbox(
        "Choose a page:", ["🤖 AI Chat Assistant", "📊 Analytics Dashboard", "ℹ️ About"]
    )

    # Page routing
    if page == "🤖 AI Chat Assistant":
        render_chat_page()
    elif page == "📊 Analytics Dashboard":
        render_dashboard_page()
    elif page == "ℹ️ About":
        render_about_page()


def render_chat_page():
    """Render chat interface page."""
    chat_interface = ChatInterface()
    chat_interface.render_chat_interface()

    # Sidebar for chat
    with st.sidebar:
        st.markdown("---")
        chat_interface.render_sidebar()


def render_dashboard_page():
    """Render analytics dashboard page."""
    dashboard = AnalyticsDashboard()
    dashboard.run()


def render_about_page():
    """Render about page."""
    st.title("ℹ️ About CrediTrust AI Platform")

    st.markdown(
        """
    ## 🎯 Mission
    CrediTrust's AI Platform transforms customer complaint data into actionable insights, 
    empowering teams to proactively address customer pain points across our financial services.
    
    ## 🚀 Features
    
    ### 🤖 AI Chat Assistant
    - **Natural Language Queries**: Ask questions in plain English
    - **Intelligent Retrieval**: Semantic search across complaint narratives
    - **Source Attribution**: View original complaints that inform each answer
    - **Multi-Product Analysis**: Compare issues across Credit Cards, Loans, BNPL, Savings, and Transfers
    
    ### 📊 Analytics Dashboard
    - **Real-time Metrics**: Key performance indicators and trends
    - **Product Analysis**: Complaint distribution across financial products
    - **Geographic Insights**: State-wise complaint patterns
    - **Temporal Trends**: Time-based analysis of complaint volumes
    
    ## 🏗️ Technical Architecture
    
    ### RAG Pipeline
    - **Text Processing**: Advanced chunking and cleaning of complaint narratives
    - **Embeddings**: Sentence-transformers for semantic understanding
    - **Vector Store**: FAISS for efficient similarity search
    - **Generation**: LLM-powered response generation with source attribution
    
    ### Data Processing
    - **Source**: Consumer Financial Protection Bureau (CFPB) complaint database
    - **Filtering**: Focus on CrediTrust's 5 core financial products
    - **Cleaning**: Automated text preprocessing and normalization
    - **Indexing**: Optimized vector storage for fast retrieval
    
    ## 📈 Business Impact
    
    ### For Product Managers
    - Identify emerging issues in minutes, not days
    - Data-driven product improvement decisions
    - Proactive issue resolution
    
    ### For Support Teams
    - Quick access to similar complaint patterns
    - Evidence-based response strategies
    - Reduced resolution time
    
    ### For Compliance Teams
    - Automated trend detection
    - Risk pattern identification
    - Regulatory reporting insights
    
    ## 🛠️ Getting Started
    
    1. **Explore the Chat**: Ask questions about customer complaints
    2. **Review Analytics**: Examine complaint trends and patterns
    3. **Dive Deep**: Use source attribution to understand root causes
    
    ## 📞 Support
    For technical support or feature requests, contact the Data & AI Engineering team.
    """
    )

    # System status
    st.markdown("---")
    st.subheader("🔧 System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if Path("vector_store/faiss_index").exists():
            st.success("✅ Vector Store: Ready")
        else:
            st.error("❌ Vector Store: Not Built")

    with col2:
        if Path("filtered_data.parquet").exists():
            st.success("✅ Data: Available")
        else:
            st.error("❌ Data: Missing")

    with col3:
        st.info("🔄 Last Updated: Today")


if __name__ == "__main__":
    main()
