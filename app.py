# Entry point for the Gradio/Streamlit app
import streamlit as st
from src.rag_pipeline import RAGPipeline

class finance_complaint_ai:
    def __init__(self):
        self.rag = RAGPipeline()
        self.user_input = ""

    def run(self):
        st.title("CrediTrust Complaint Insight Bot")
        st.markdown("Ask a question based on customer complaints.")

        self.user_input = st.text_input("Your question:")

        if st.button("Submit"):
            with st.spinner("Thinking..."):
                answer, sources = self.rag.answer(self.user_input)
                st.write("### Answer")
                st.write(answer)

                st.write("### Sources")
                for idx, meta in sources:
                    st.markdown(f"- **Product**: {meta['product']} | **Complaint ID**: {meta['id']}")

        if st.button("Clear"):
            st.experimental_rerun()

def main():
    app = finance_complaint_ai()
    app.run()

if __name__ == "__main__":
    main()