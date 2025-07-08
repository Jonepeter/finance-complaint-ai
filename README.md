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

```markdown
. 📂 finance-complaint-ai
├── 📄 README.md
├── 📄 app.py
└── 📂 data/
└── 📂 notebooks/
│  ├── 📄 01_eda_preprocessing.ipynb
│  ├── 📄 chunking_embeding.ipynb
└── 📂 outputs/
│  ├── 📄 output1.png
└── 📂 reports/
├── 📄 requirements.txt
└── 📂 src/
│  ├── 📄 __init__.py
│  └── 📂 __pycache__/
│  ├── 📄 chunk_and_embed.py
│  ├── 📄 data_preprocessing.py
│  ├── 📄 evaluation.py
│  ├── 📄 rag_pipeline.py
│  ├── 📄 utils.py
└── 📂 tests/
│  ├── 📄 test_rag_pipeline.py
└── 📂 vector_store/

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

## Workflow

- Data → Chunking → Embeddings → FAISS → Retrieval → LLM Response

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

## Setup and Installation

Follow these steps to get the project up and running:

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone [https://github.com/your-username/finance-complaint-ai.git](https://github.com/Jonepeter/finance-complaint-ai.git)
    cd finance-complaint-ai/rag_chatbot
    ```
    (Note: This is a placeholder for `your-username/finance-complaint-ai` as the original repo was inaccessible. If you are creating a new repo, start from `rag_chatbot` directory.)

2. **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Chatbot

After completing the setup:

1. **Ensure your virtual environment is active.**
2. **Run the main application script:**

```bash
    python app.py
```

This will set up the RAG pipeline, load/process your data, and launch the Gradio web interface. Open the URL provided by Gradio (usually `http://127.0.0.1:7860`) in your web browser to interact with the chatbot.

## Customization and Extension

**Data:** Add more `.txt` files to the `data/` directory to expand the chatbot's knowledge base.
**Text Processing:** Adjust `chunk_size` and `chunk_overlap` in `text_processor.py` for different document types.
**Embeddings:** Change the `EMBEDDING_MODEL` in `.env` to experiment with different embedding models.
**LLM:** Switch the `LLM_MODEL` in `.env` to use OpenAI's models (with API key) or a local Ollama model.
**Prompt Engineering:** Modify the `PROMPT_TEMPLATE` in `.env` or directly in `rag_pipeline.py` to refine the chatbot's response style and accuracy.
**Vector Store:** While ChromaDB is used, the `vector_store_manager.py` can be extended to support other vector databases (e.g., FAISS, Pinecone, Qdrant).
**Gradio UI:** Customize the `chatbot_interface.py` to change the layout, add more components, or modify the theme of the Gradio application.

---

## Onboarding & Contributing

- New contributors should start by reading this README and exploring the `notebooks/` for data exploration and prototyping.
- Follow the file naming conventions for any new scripts or notebooks.
- Add docstrings and comments to your code to help others understand your logic.
- For major changes, open an issue or pull request for discussion.
- If you have questions, open an issue or contact the maintainers.

---
