"""
RAG pipeline orchestration for finance-complaint-ai.
This module contains the main logic for retrieval-augmented generation using embeddings and vector stores.
"""

# Functions will be implemented here. 
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.load_local(
    "../vector_store/creditrust_faiss_index",
    embedding_model
)

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 1000}
)

# Prompt template
prompt_template = """You are a financial analyst assistant for CrediTrust. 
Your task is to answer questions about customer complaints. 
Use only the following retrieved complaint excerpts to formulate your answer. 
If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer in a concise, professional manner:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

def ask_question(question):
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }