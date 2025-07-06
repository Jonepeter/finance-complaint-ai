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
- Gradio

## Project Structure

```
finance-complaint-ai/
├── app.py                  # Gradio/Streamlit app entry point
├── data/                   # Raw and processed data files
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── outputs/                # Output files (plots, images, etc.)
├── reports/                # Reports and documentation
├── requirements.txt        # Python dependencies
├── src/                    # Source code for data processing, chunking, embedding, RAG logic, and utilities
│   ├── __init__.py
│   ├── chunk_and_embed.py
│   ├── data_preprocessing.py
│   ├── embedding.py
│   ├── evaluation.py
│   ├── rag_pipeline.py
│   └── utils.py
├── tests/                  # Unit and integration tests
├── vector_store/           # Persisted vector database files
└── .github/                # GitHub Actions workflows and settings
```

- **app.py**: Main entry point for the chatbot web app (Gradio/Streamlit).
- **data/**: Contains raw and processed datasets.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and prototyping.
- **outputs/**: Stores generated outputs such as plots and images.
- **reports/**: Project reports and additional documentation.
- **requirements.txt**: List of Python dependencies.
- **src/**: All core source code modules for data processing, chunking, embedding, RAG pipeline, and utilities.
- **tests/**: Unit and integration tests for the codebase.
- **vector_store/**: Directory for storing FAISS or other vector database files.
- **.github/**: Contains GitHub Actions workflows and repository settings.

## ⚡ Quickstart

### Prerequisites

- Python 3.13.1

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

---

## How to Run Each Part

This project is modular. Here's how to execute each major part:

### 1. Data Preprocessing
- Prepare your data in the `data/` directory (e.g., `filtered_data.parquet` or `filtered_data.csv`).
- If you need to clean or preprocess data, use or adapt scripts in `src/data_preprocessing.py`.

### 2. Chunking and Embedding
- Run the chunking and embedding pipeline to build the vector store:

```bash
python src/chunk_and_embed.py
```
- This will process the data and create a FAISS vector store in `vector_store/`.

### 3. Running the App
- Start the chatbot web app:

```bash
python app.py
```
- This launches a Gradio interface for interacting with the RAG-powered chatbot.

### 4. Running Tests
- To run all tests:

```bash
pytest
```
- Place your test scripts in the `tests/` directory.

---

## File Naming Conventions
- Use lowercase letters and underscores for Python files (e.g., `data_preprocessing.py`, `chunk_and_embed.py`).
- Notebooks should be named descriptively (e.g., `01_eda_preprocessing.ipynb`).
- Output files and reports should be named to reflect their content or purpose.

---

## Example Usage

**Ask a question using the RAG pipeline in Python:**

```python
from src.rag_pipeline import ask_question
result = ask_question("What are the most common complaints about credit cards?")
print(result['answer'])
```

---

## Onboarding & Contributing

- New contributors should start by reading this README and exploring the `notebooks/` for data exploration and prototyping.
- Follow the file naming conventions for any new scripts or notebooks.
- Add docstrings and comments to your code to help others understand your logic.
- For major changes, open an issue or pull request for discussion.
- If you have questions, open an issue or contact the maintainers.

---
