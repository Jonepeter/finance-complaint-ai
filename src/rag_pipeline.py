from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import pipeline

class RAGPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2', llm_model='mistralai/Mistral-7B-Instruct-v0.1'):
        self.embedder = SentenceTransformer(model_name)
        self.index = faiss.read_index("vector_store/faiss.index")
        with open("vector_store/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        self.llm = pipeline("text-generation", model=llm_model)

    def retrieve(self, query, k=5):
        vec = self.embedder.encode([query])
        D, I = self.index.search(vec, k)
        results = [(I[0][i], self.metadata[I[0][i]]) for i in range(k)]
        return results

    def generate_prompt(self, context, question):
        return f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
                Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information. 
                Context:{context}

                Question: {question}
                Answer: """

    def answer(self, question):
        results = self.retrieve(question)
        context = "\n\n".join([self.metadata[i]['product'] + ": " + documents[i] for i, _ in results])
        prompt = self.generate_prompt(context, question)
        return self.llm(prompt, max_new_tokens=150)[0]['generated_text'], results