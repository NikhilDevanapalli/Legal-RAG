import argparse
from pathlib import Path
from typing import Optional

from config import CORPUS_SPLIT, LLM_MODEL, TOP_N_EMBEDDINGS, EMBEDDING_MODEL
from agent import get_rag_answer
from evaluate_questions import (
    get_next_output_path,
    load_vector_store_and_agent,
    load_qa_split,
    resolve_end_index,
)

try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
    )
    from ragas import evaluate
    from ragas.llms import llm_factory
    from datasets import Dataset
except ImportError:
    raise ImportError(
        "ragas is required for this evaluation framework. "
        "Install it with: pip install ragas"
    )

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai is required for RAGAS evaluation. "
        "Install it with: pip install openai"
    )

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
    Workbook = None


def write_ragas_excel(evaluation_results, output_path: Path, metadata: dict):
    """Write RAGAS evaluation results to Excel with metrics and metadata."""
    if Workbook is None:
        raise ImportError(
            "openpyxl is required to write Excel files. Install it with: pip install openpyxl"
        )

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "RAGAS Evaluation"

    # Add metadata header
    sheet.append(["Configuration"])
    for key, value in metadata.items():
        sheet.append([key, value])

    # Add blank row
    sheet.append([])

    # Add metrics summary
    sheet.append(["Metrics Summary"])
    if "overall_metrics" in evaluation_results:
        for metric_name, metric_value in evaluation_results["overall_metrics"].items():
            sheet.append([metric_name, f"{metric_value:.4f}"])

    # Add blank row
    sheet.append([])

    # Add detailed results header
    sheet.append([
        "Index",
        "Question",
        "Retrieved Context",
        "Model Answer",
        "Actual Answer",
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
    ])

    # Add detailed results
    if "per_sample_results" in evaluation_results:
        for result in evaluation_results["per_sample_results"]:
            sheet.append([
                result.get("index"),
                result.get("question"),
                result.get("context", "")[:100] + "..." if result.get("context") else "",  # Truncate for readability
                result.get("answer"),
                result.get("reference"),
                f"{result.get('faithfulness', 0):.4f}",
                f"{result.get('answer_relevancy', 0):.4f}",
                f"{result.get('context_precision', 0):.4f}",
            ])

    workbook.save(output_path)




def evaluate_with_ragas(
    model: str,
    split: str,
    start_index: int,
    limit: Optional[int],
    output_path: Path,
    top_k: int,
    embedding_model: str,
    vector_store_type: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Evaluate QA with RAGAS metrics and export to Excel."""
    agent, retriever = load_vector_store_and_agent(
        model=model,
        top_k=top_k,
        embedding_model=embedding_model,
        vector_store_type=vector_store_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    dataset = load_qa_split(split)
    total_questions = len(dataset["question"])
    end_index = resolve_end_index(total_questions, start_index, limit)

    print(
        f"Evaluating questions {start_index}..{end_index - 1} from split '{split}' using model '{model}'."
    )

    # Collect data for RAGAS evaluation
    questions = []
    answers = []
    ground_truths = []
    contexts_list = []
    indices = []

    for idx in range(start_index, end_index):
        question = dataset["question"][idx]
        actual_answer = dataset["answer"][idx]

        print(f"[{idx}] Generating answer and retrieving context...")
        answer = get_rag_answer(agent, question)
        retrieved_docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        questions.append(question)
        answers.append(answer)
        ground_truths.append(actual_answer)
        contexts_list.append(context)
        indices.append(idx)

    # Prepare dataset for RAGAS
    print("\nRunning RAGAS evaluation...")
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": [[ctx] for ctx in contexts_list],  # RAGAS expects list of context lists
        "ground_truth": ground_truths,
    })

    # Run RAGAS evaluation with latest API
    try:
        openai_client = OpenAI()
        ragas_llm = llm_factory(model, client=openai_client)
        
        # Set LLM for metrics
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        context_precision.llm = ragas_llm
        
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
        ]
        
        print("Metrics initialized. Starting evaluation...")
        ragas_results = evaluate(
            ragas_dataset,
            metrics=metrics,
        )
    except Exception as e:
        print(f"Warning: RAGAS evaluation encountered an issue: {e}")
        print("Returning basic results without RAGAS metrics.")
        ragas_results = None

    # Prepare results
    per_sample_results = []
    if ragas_results:
        # Extract scores from RAGAS results
        # ragas_results.scores is a list of dicts with metric scores for each sample
        for i, idx in enumerate(indices):
            # Get scores from the scores list
            score_dict = ragas_results.scores[i] if i < len(ragas_results.scores) else {}
            
            faithfulness_score = float(score_dict.get("faithfulness", 0))
            answer_relevancy_score = float(score_dict.get("answer_relevancy", 0))
            context_precision_score = float(score_dict.get("context_precision", 0))
            
            per_sample_results.append({
                "index": idx,
                "question": questions[i],
                "context": contexts_list[i],
                "answer": answers[i],
                "reference": ground_truths[i],
                "faithfulness": faithfulness_score,
                "answer_relevancy": answer_relevancy_score,
                "context_precision": context_precision_score,
            })

        # Calculate overall metrics
        faithfulness_scores = [float(s.get("faithfulness", 0)) for s in ragas_results.scores]
        answer_relevancy_scores = [float(s.get("answer_relevancy", 0)) for s in ragas_results.scores]
        context_precision_scores = [float(s.get("context_precision", 0)) for s in ragas_results.scores]
        
        overall_metrics = {
            "Avg Faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            "Avg Answer Relevancy": sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0,
            "Avg Context Precision": sum(context_precision_scores) / len(context_precision_scores) if context_precision_scores else 0,
        }
    else:
        # Create results without RAGAS metrics
        for i, idx in enumerate(indices):
            per_sample_results.append({
                "index": idx,
                "question": questions[i],
                "context": contexts_list[i],
                "answer": answers[i],
                "reference": ground_truths[i],
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": 0,
            })
        overall_metrics = {
            "Avg Faithfulness": 0,
            "Avg Answer Relevancy": 0,
            "Avg Context Precision": 0,
        }

    # Prepare metadata
    metadata = {
        "LLM Model": model,
        "Embedding Model": embedding_model,
        "Vector Store": vector_store_type,
        "Top-K": top_k,
        "Chunk Size": chunk_size,
        "Chunk Overlap": chunk_overlap,
        "Dataset Split": split,
        "Start Index": start_index,
        "Number of Questions": len(questions),
    }

    # Write results
    evaluation_results = {
        "overall_metrics": overall_metrics,
        "per_sample_results": per_sample_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path = get_next_output_path(output_path)
    write_ragas_excel(evaluation_results, final_output_path, metadata)
    print(f"\nSaved RAGAS evaluation results to {final_output_path}")
    print("\nMetrics Summary:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a QA dataset with RAGAS metrics and export results to Excel."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model to use (default: {LLM_MODEL}).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Embedding model to use (default: {EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=CORPUS_SPLIT,
        help=f"Dataset split to evaluate (default: {CORPUS_SPLIT}).",
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="faiss",
        choices=["faiss", "chroma"],
        help="Vector store type to use (default: faiss).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Text chunk size for vector store (default: 800).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Text chunk overlap for vector store (default: 150).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based question index to start evaluation from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of questions to evaluate. Use 0 to evaluate all questions.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_N_EMBEDDINGS,
        help=f"Number of retrieved documents per query (default: {TOP_N_EMBEDDINGS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ragas_evaluation_results.xlsx"),
        help="Path to the Excel output file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    limit = None if args.limit == 0 else args.limit
    evaluate_with_ragas(
        model=args.model,
        split=args.split,
        start_index=args.start_index,
        limit=limit,
        output_path=args.output,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        vector_store_type=args.vector_store,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
