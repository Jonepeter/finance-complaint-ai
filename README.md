# Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) powered chatbot to turn customer feedback into actionable insights for CrediTrust Financial.

## Project Introduction

This project aims to help CrediTrust Financial transform large volumes of customer complaints and feedback into actionable business insights. By combining advanced Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) techniques, we have developed an intelligent chatbot capable of understanding, categorizing, and summarizing customer feedback. The solution enables the organization to quickly identify key issues, trends, and opportunities for improvement, ultimately enhancing customer satisfaction and operational decision-making.

## Key Features

- **Automated Complaint Analysis:** Uses NLP to automatically categorize and summarize customer complaints.
- **Retrieval-Augmented Generation (RAG):** Combines retrieval of relevant complaint data with generative AI for accurate, context-aware responses.
- **Actionable Insights:** Surfaces trends, recurring issues, and opportunities for business improvement.
- **User-Friendly Chatbot Interface:** Interact with the system via a simple web interface (Gradio/Streamlit).
- **Customizable & Extensible:** Modular codebase for easy adaptation to other domains or data sources.

## Usage Scenarios

- **Customer Service Teams:** Quickly identify and address common pain points.
- **Product Managers:** Discover feature requests and product issues from real customer feedback.
- **Business Analysts:** Generate reports on complaint trends and root causes.

## Technologies Used

- Python, Pandas, NumPy, scikit-learn
- Sentence Transformers, FAISS, ChromaDB
- LangChain, OpenAI API
- Streamlit, Gradio

## Project Structure

- `data/`: Raw and processed data files
- `notebooks/`: Jupyter notebooks for EDA and prototyping
- `src/`: Source code for data processing, chunking, embedding, RAG logic, and utilities
- `vector_store/`: Persisted vector database files
- `app.py`: Gradio/Streamlit app entry point
- `reports/`: Reports and documentation
- `tests/`: Unit and integration tests

## Setup

1. Clone the repository
2. Install dependencies:

```bash

   pip install -r requirements.txt
```

3.Run the app:

```bash
   python app.py
```

## Getting Help

For questions or support, please open an issue or contact the project maintainers.
