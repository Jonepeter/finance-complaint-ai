"""RAG pipeline evaluation utilities.

This module provides utilities for evaluating the performance of a Retrieval-Augmented Generation (RAG) pipeline.
It includes the RAGEvaluator class, which can run a set of evaluation questions, score the responses, and generate a markdown report.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from src.rag.rag_pipeline import RAGPipeline
except ImportError:
    # Fallback for direct execution
    from rag_pipeline import RAGPipeline


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance by running a set of predefined questions,
    scoring the answers, and generating a summary report.

    Attributes:
        rag_pipeline (RAGPipeline): The RAG pipeline instance to evaluate.
        evaluation_questions (List[str]): List of questions for evaluation.
    """

    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the evaluator with a RAG pipeline instance.

        Args:
            rag_pipeline (RAGPipeline): The RAG pipeline to evaluate.
        """
        self.rag_pipeline = rag_pipeline
        self.evaluation_questions = [
            "Why are people unhappy with BNPL services?",
            "What are the main issues with credit cards?",
            "What problems do customers face with personal loans?",
            "What are common complaints about savings accounts?",
            "What issues do people have with money transfers?",
            "Which product has the most billing disputes?",
            "What are the top 3 complaint categories?",
            "How do customers feel about customer service?",
            "What are the most frequent payment-related issues?",
            "Which states have the most complaints?",
        ]

    def evaluate_question(self, question: str) -> Dict[str, Any]:
        """
        Evaluate a single question using the RAG pipeline.

        Args:
            question (str): The question to evaluate.

        Returns:
            Dict[str, Any]: Dictionary containing question, answer, sources, confidence, quality_score, and num_sources.
        """
        try:
            result = self.rag_pipeline.generate_response(question)
        except Exception as e:
            # Handle errors gracefully and return a default result
            result = {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "confidence": 0.0,
            }

        # Simple quality scoring based on response characteristics
        try:
            quality_score = self._calculate_quality_score(result)
        except Exception as e:
            quality_score = 1.0  # Minimum score if scoring fails

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", [])[:2],  # Top 2 sources
            "confidence": result.get("confidence", 0.0),
            "quality_score": quality_score,
            "num_sources": len(result.get("sources", [])),
        }

    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate a quality score (1-5) for a response based on its characteristics.

        Args:
            result (Dict[str, Any]): The result dictionary from the RAG pipeline.

        Returns:
            float: The quality score (1.0 to 5.0).
        """
        score = 1.0

        # Check if answer is meaningful
        answer = result.get("answer", "").lower()
        if len(answer) > 50 and not any(
            phrase in answer
            for phrase in ["don't have enough information", "error", "try again"]
        ):
            score += 1.0

        # Check confidence
        confidence = result.get("confidence", 0.0)
        if confidence > 0.7:
            score += 1.0
        elif confidence > 0.5:
            score += 0.5

        # Check number of sources
        num_sources = len(result.get("sources", []))
        if num_sources >= 3:
            score += 1.0
        elif num_sources >= 1:
            score += 0.5

        # Check answer specificity
        if any(
            word in answer for word in ["specific", "particular", "mainly", "primarily"]
        ):
            score += 0.5

        return min(score, 5.0)

    def run_evaluation(self) -> pd.DataFrame:
        """
        Run the complete evaluation over all predefined questions.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results for each question.
        """
        results = []

        print("Running RAG evaluation...")
        for i, question in enumerate(self.evaluation_questions, 1):
            print(
                f"Evaluating question {i}/{len(self.evaluation_questions)}: {question}"
            )
            try:
                result = self.evaluate_question(question)
            except Exception as e:
                # If evaluation fails, log the error and continue
                print(f"Error evaluating question '{question}': {e}")
                result = {
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "confidence": 0.0,
                    "quality_score": 1.0,
                    "num_sources": 0,
                }
            results.append(result)

        # Create evaluation dataframe
        try:
            eval_df = pd.DataFrame(results)
        except Exception as e:
            print(f"Error creating evaluation DataFrame: {e}")
            eval_df = pd.DataFrame()

        # Add summary statistics
        try:
            avg_quality = eval_df["quality_score"].mean()
            avg_confidence = eval_df["confidence"].mean()
        except Exception:
            avg_quality = 0.0
            avg_confidence = 0.0

        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Average Quality Score: {avg_quality:.2f}/5.0")
        print(f"Average Confidence: {avg_confidence:.2f}")
        try:
            print(
                f"Questions with Quality >= 3.0: {(eval_df['quality_score'] >= 3.0).sum()}/{len(eval_df)}"
            )
        except Exception:
            print("Could not compute high quality response count.")

        return eval_df

    def generate_evaluation_report(self, eval_df: pd.DataFrame) -> str:
        """
        Generate a markdown evaluation report from the evaluation DataFrame.

        Args:
            eval_df (pd.DataFrame): The evaluation results DataFrame.

        Returns:
            str: Markdown-formatted evaluation report.
        """
        report = "# RAG Pipeline Evaluation Report\n\n"
        try:
            report += "## Summary Statistics\n\n"
            report += f"- **Average Quality Score**: {eval_df['quality_score'].mean():.2f}/5.0\n"
            report += f"- **Average Confidence**: {eval_df['confidence'].mean():.2f}\n"
            report += f"- **High Quality Responses**: {(eval_df['quality_score'] >= 3.0).sum()}/{len(eval_df)}\n\n"
        except Exception as e:
            report += f"Error computing summary statistics: {e}\n\n"

        report += "## Detailed Results\n\n"
        report += (
            "| Question | Generated Answer | Quality Score | Confidence | Comments |\n"
        )
        report += (
            "|----------|------------------|---------------|------------|----------|\n"
        )

        try:
            for _, row in eval_df.iterrows():
                answer_preview = (
                    row["answer"][:100] + "..."
                    if len(row["answer"]) > 100
                    else row["answer"]
                )
                quality_comment = self._get_quality_comment(row["quality_score"])

                report += f"| {row['question']} | {answer_preview} | {row['quality_score']:.1f} | {row['confidence']:.2f} | {quality_comment} |\n"
        except Exception as e:
            report += f"| Error generating detailed results: {e} |||||\n"

        report += "\n## Analysis\n\n"
        try:
            report += self._generate_analysis(eval_df)
        except Exception as e:
            report += f"Error generating analysis: {e}\n"

        return report

    def _get_quality_comment(self, score: float) -> str:
        """
        Get a qualitative comment based on the quality score.

        Args:
            score (float): The quality score.

        Returns:
            str: Qualitative comment.
        """
        if score >= 4.0:
            return "Excellent"
        elif score >= 3.0:
            return "Good"
        elif score >= 2.0:
            return "Fair"
        else:
            return "Needs improvement"

    def _generate_analysis(self, eval_df: pd.DataFrame) -> str:
        """
        Generate an analysis section for the evaluation report.

        Args:
            eval_df (pd.DataFrame): The evaluation results DataFrame.

        Returns:
            str: Markdown-formatted analysis section.
        """
        analysis = ""

        # Best performing questions
        try:
            best_questions = eval_df.nlargest(3, "quality_score")
            analysis += "### Best Performing Questions\n\n"
            for _, row in best_questions.iterrows():
                analysis += (
                    f"- **{row['question']}** (Score: {row['quality_score']:.1f})\n"
                )
        except Exception as e:
            analysis += f"Error finding best performing questions: {e}\n"

        # Areas for improvement
        try:
            worst_questions = eval_df.nsmallest(3, "quality_score")
            analysis += "\n### Areas for Improvement\n\n"
            for _, row in worst_questions.iterrows():
                analysis += (
                    f"- **{row['question']}** (Score: {row['quality_score']:.1f})\n"
                )
        except Exception as e:
            analysis += f"Error finding areas for improvement: {e}\n"

        analysis += "\n### Recommendations\n\n"
        try:
            if eval_df["quality_score"].mean() < 3.0:
                analysis += (
                    "- Consider improving the embedding model or chunking strategy\n"
                )
                analysis += "- Enhance the prompt template for better LLM responses\n"

            if eval_df["confidence"].mean() < 0.5:
                analysis += "- Review the similarity threshold for retrieval\n"
                analysis += (
                    "- Consider expanding the dataset or improving text preprocessing\n"
                )
        except Exception as e:
            analysis += f"Error generating recommendations: {e}\n"

        return analysis
