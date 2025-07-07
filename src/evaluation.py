"""
Evaluation metrics and functions for the finance-complaint-ai project.
Use this module to evaluate model and pipeline performance.
"""

# Add evaluation functions below as needed 
import pandas as pd

def qualitative_evaluation(rag_pipeline, questions, top_k_sources=2):
    """
    Run qualitative evaluation on a list of representative questions using the RAG pipeline.

    Args:
        rag_pipeline: Callable that takes a question and returns a dict with keys:
            - 'answer': generated answer (str)
            - 'sources': list of retrieved source strings (list of str)
        questions: List of questions (str) to evaluate.
        top_k_sources: Number of retrieved sources to display per question.

    Returns:
        pd.DataFrame: Evaluation table with columns:
            - Question
            - Generated Answer
            - Retrieved Sources
            - Quality Score
            - Comments/Analysis
    """
    results = []
    print("Qualitative Evaluation: Please rate each answer (1-5) and provide comments.\n")
    for idx, question in enumerate(questions, 1):
        output = rag_pipeline(question)
        answer = output.get('answer', '')
        sources = output.get('sources', [])[:top_k_sources]
        print(f"\nQ{idx}: {question}")
        print(f"Generated Answer:\n{answer}\n")
        print("Retrieved Sources:")
        for i, src in enumerate(sources, 1):
            print(f"  [{i}] {src}")
        quality = input("Quality Score (1-5): ")
        comments = input("Comments/Analysis: ")
        results.append({
            "Question": question,
            "Generated Answer": answer,
            "Retrieved Sources": "\n\n".join(sources),
            "Quality Score": quality,
            "Comments/Analysis": comments
        })
    df = pd.DataFrame(results, columns=[
        "Question", "Generated Answer", "Retrieved Sources", "Quality Score", "Comments/Analysis"
    ])
    return df

def evaluation_table_to_markdown(df, filepath=None):
    """
    Convert the evaluation DataFrame to a Markdown table and optionally save to a file.

    Args:
        df: pd.DataFrame as returned by qualitative_evaluation.
        filepath: Optional path to save the Markdown table.

    Returns:
        str: Markdown table as a string.
    """
    markdown = df.to_markdown(index=False)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)
    return markdown
