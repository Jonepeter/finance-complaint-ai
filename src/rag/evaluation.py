"""RAG pipeline evaluation utilities."""

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
    """Evaluates RAG pipeline performance."""

    def __init__(self, rag_pipeline: RAGPipeline):
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
        """Evaluate a single question."""
        result = self.rag_pipeline.generate_response(question)

        # Simple quality scoring based on response characteristics
        quality_score = self._calculate_quality_score(result)

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"][:2],  # Top 2 sources
            "confidence": result["confidence"],
            "quality_score": quality_score,
            "num_sources": len(result["sources"]),
        }

    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score (1-5) based on response characteristics."""
        score = 1.0

        # Check if answer is meaningful
        answer = result["answer"].lower()
        if len(answer) > 50 and not any(
            phrase in answer
            for phrase in ["don't have enough information", "error", "try again"]
        ):
            score += 1.0

        # Check confidence
        if result["confidence"] > 0.7:
            score += 1.0
        elif result["confidence"] > 0.5:
            score += 0.5

        # Check number of sources
        if len(result["sources"]) >= 3:
            score += 1.0
        elif len(result["sources"]) >= 1:
            score += 0.5

        # Check answer specificity
        if any(
            word in answer for word in ["specific", "particular", "mainly", "primarily"]
        ):
            score += 0.5

        return min(score, 5.0)

    def run_evaluation(self) -> pd.DataFrame:
        """Run complete evaluation."""
        results = []

        print("Running RAG evaluation...")
        for i, question in enumerate(self.evaluation_questions, 1):
            print(
                f"Evaluating question {i}/{len(self.evaluation_questions)}: {question}"
            )
            result = self.evaluate_question(question)
            results.append(result)

        # Create evaluation dataframe
        eval_df = pd.DataFrame(results)

        # Add summary statistics
        avg_quality = eval_df["quality_score"].mean()
        avg_confidence = eval_df["confidence"].mean()

        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Average Quality Score: {avg_quality:.2f}/5.0")
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(
            f"Questions with Quality >= 3.0: {(eval_df['quality_score'] >= 3.0).sum()}/{len(eval_df)}"
        )

        return eval_df

    def generate_evaluation_report(self, eval_df: pd.DataFrame) -> str:
        """Generate markdown evaluation report."""
        report = "# RAG Pipeline Evaluation Report\\n\\n"
        report += "## Summary Statistics\\n\\n"
        report += (
            f"- **Average Quality Score**: {eval_df['quality_score'].mean():.2f}/5.0\\n"
        )
        report += f"- **Average Confidence**: {eval_df['confidence'].mean():.2f}\\n"
        report += f"- **High Quality Responses**: {(eval_df['quality_score'] >= 3.0).sum()}/{len(eval_df)}\\n\\n"

        report += "## Detailed Results\\n\\n"
        report += (
            "| Question | Generated Answer | Quality Score | Confidence | Comments |\\n"
        )
        report += (
            "|----------|------------------|---------------|------------|----------|\\n"
        )

        for _, row in eval_df.iterrows():
            answer_preview = (
                row["answer"][:100] + "..."
                if len(row["answer"]) > 100
                else row["answer"]
            )
            quality_comment = self._get_quality_comment(row["quality_score"])

            report += f"| {row['question']} | {answer_preview} | {row['quality_score']:.1f} | {row['confidence']:.2f} | {quality_comment} |\\n"

        report += "\\n## Analysis\\n\\n"
        report += self._generate_analysis(eval_df)

        return report

    def _get_quality_comment(self, score: float) -> str:
        """Get quality comment based on score."""
        if score >= 4.0:
            return "Excellent"
        elif score >= 3.0:
            return "Good"
        elif score >= 2.0:
            return "Fair"
        else:
            return "Needs improvement"

    def _generate_analysis(self, eval_df: pd.DataFrame) -> str:
        """Generate analysis section."""
        analysis = ""

        # Best performing questions
        best_questions = eval_df.nlargest(3, "quality_score")
        analysis += "### Best Performing Questions\\n\\n"
        for _, row in best_questions.iterrows():
            analysis += (
                f"- **{row['question']}** (Score: {row['quality_score']:.1f})\\n"
            )

        # Areas for improvement
        worst_questions = eval_df.nsmallest(3, "quality_score")
        analysis += "\\n### Areas for Improvement\\n\\n"
        for _, row in worst_questions.iterrows():
            analysis += (
                f"- **{row['question']}** (Score: {row['quality_score']:.1f})\\n"
            )

        analysis += "\\n### Recommendations\\n\\n"
        if eval_df["quality_score"].mean() < 3.0:
            analysis += (
                "- Consider improving the embedding model or chunking strategy\\n"
            )
            analysis += "- Enhance the prompt template for better LLM responses\\n"

        if eval_df["confidence"].mean() < 0.5:
            analysis += "- Review the similarity threshold for retrieval\\n"
            analysis += (
                "- Consider expanding the dataset or improving text preprocessing\\n"
            )

        return analysis
