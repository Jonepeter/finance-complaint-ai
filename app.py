# Entry point for the Gradio/Streamlit app
import streamlit as st
from src.rag_pipeline import RAGPipeline

st.title("CrediTrust Complaint Insight Bot")
st.markdown("Ask a question based on customer complaints.")

user_input = st.text_input("Your question:")
rag = RAGPipeline()

if st.button("Submit"):
    with st.spinner("Thinking..."):
        answer, sources = rag.answer(user_input)
        st.write("### Answer")
        st.write(answer)

        st.write("### Sources")
        for idx, meta in sources:
            st.markdown(f"- **Product**: {meta['product']} | **Complaint ID**: {meta['id']}")

if st.button("Clear"):
    st.experimental_rerun()