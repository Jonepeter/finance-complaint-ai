# 🏦 CrediTrust Financial - Intelligent Complaint Analysis System

A comprehensive RAG-powered AI platform that transforms customer complaint data into actionable business insights for financial services.

## 🎯 Project Overview

CrediTrust Financial's AI platform empowers internal teams to quickly understand customer pain points across five major financial products:
- Credit Cards
- Personal Loans  
- Buy Now, Pay Later (BNPL)
- Savings Accounts
- Money Transfers

### Key Features
- **🤖 AI Chat Assistant**: Natural language queries with source attribution
- **🔍 Semantic Search**: Advanced RAG pipeline for intelligent retrieval
- **📈 Business Intelligence**: Proactive issue identification and resolution

## 🏗️ Project Structure

```bash
CrediTrust-AI-Platform/
├── src/
│   ├── rag/                    # RAG pipeline components
│   │   ├── text_processor.py   # Text chunking and cleaning
│   │   ├── embeddings.py       # Vector store management
│   │   ├── rag_pipeline.py     # Complete RAG implementation
│   │   └── evaluation.py       # Performance evaluation
│   ├── dashboard/              # User interfaces
│   │   ├── chat_interface.py   # Streamlit chat app
│   │   └── analytics_dashboard.py # Analytics dashboard
│   └── utils/
│       └── config_loader.py    # Configuration management
├── notebooks/
│   └── eda/
│       └── 01_exploratory_data_analysis.ipynb
├── data/
│   ├── raw/                    # Original data files
│   └── processed/              # Cleaned datasets
├── config/
│   └── config.yaml            # System configuration
├── vector_store/              # FAISS index and metadata
├── models/                    # Model artifacts and evaluation
├── docs/                      # Documentation
├── app.py                     # Main Streamlit application
├── build_vector_store.py      # Vector store builder
├── evaluate_rag.py           # RAG evaluation script
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd CrediTrust-AI-Platform

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Ensure the data is in the root directory with complaint data.

### 3. Build Vector Store

```bash
python build_vector_store.py
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📊 Usage Guide

### AI Chat Assistant
1. Navigate to the "🤖 AI Chat Assistant" page
2. Ask questions like:
   - "Why are people unhappy with BNPL services?"
   - "What are the main credit card issues?"
   - "Which product has the most billing disputes?"
3. Review generated answers with source attribution
4. Explore complaint sources for deeper insights


## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Embedding models and parameters
- Chunking strategies
- Vector store settings

## 📈 Evaluation

Run comprehensive RAG evaluation:

```bash
python evaluate_rag.py
```

Results saved to:
- `models/evaluation_results.csv` - Detailed metrics
- `docs/evaluation_report.md` - Analysis report

## 🛠️ Development

### Adding New Features
1. Follow the existing code structure
2. Update configuration in `config.yaml`
3. Add tests for new functionality
4. Update documentation

### Model Improvements
- Experiment with different embedding models
- Adjust chunking parameters
- Fine-tune retrieval settings
- Enhance prompt templates

## 📋 Tasks Completed

### ✅ Task 1: EDA and Preprocessing
- Comprehensive data analysis in Jupyter notebook
- Text cleaning and normalization
- Product filtering and data quality assessment
- Filtered dataset creation

### ✅ Task 2: Text Chunking and Embeddings
- Intelligent text chunking with overlap
- Sentence-transformer embeddings
- FAISS vector store implementation
- Metadata preservation for source attribution

### ✅ Task 3: RAG Pipeline and Evaluation
- Complete retrieval-augmented generation system
- Prompt engineering for financial domain
- Comprehensive evaluation framework
- Performance metrics and analysis

### ✅ Task 4: Interactive Interfaces
- Streamlit chat interface with source display
- Real-time response streaming
- Analytics dashboard with multiple visualizations
- Multi-page navigation system

### ✅ Additional: MLOps Integration
- Standardized project structure
- Configuration management
- Automated evaluation pipeline
- Documentation and deployment guides

## 🎯 Business Impact

### For Product Managers
- **Time Reduction**: Issue identification from days to minutes
- **Data-Driven Decisions**: Evidence-based product improvements
- **Proactive Resolution**: Early detection of emerging problems

### For Support Teams
- **Faster Resolution**: Quick access to similar complaint patterns
- **Consistent Responses**: Evidence-backed customer communications
- **Knowledge Sharing**: Centralized complaint intelligence

### For Compliance Teams
- **Automated Monitoring**: Continuous trend detection
- **Risk Assessment**: Pattern-based risk identification
- **Regulatory Reporting**: Streamlined compliance workflows

## 📞 Support

For technical support, feature requests, or questions:
- Contact: Data & AI Engineering Team
- Documentation: `docs/` directory
- Issues: Create GitHub issues for bug reports

## 🔄 Future Enhancements

- Real-time data ingestion pipeline
- Advanced NLP for sentiment analysis
- Predictive complaint modeling
- Integration with customer service systems
- Multi-language support for global operations

---

**Built with ❤️ by the CrediTrust Data & AI Engineering Team**
