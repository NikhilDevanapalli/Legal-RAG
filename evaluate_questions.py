import argparse
from pathlib import Path
from typing import Optional

from config import CORPUS_SPLIT, LLM_MODEL, TOP_N_EMBEDDINGS, EMBEDDING_MODEL
from data import load_qa_dataset
from retriever import create_search_retriever
from vector_store import load_or_create_vector_store
from agent import create_legal_agent, get_rag_answer

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
    Workbook = None


def write_excel(rows, output_path: Path, metadata: dict):
    if Workbook is None:
        raise ImportError(
            "openpyxl is required to write Excel files. Install it with: pip install openpyxl"
        )

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Evaluation"
    
    # Add metadata header
    row_num = 1
    sheet.append(["Configuration"])
    row_num += 1
    for key, value in metadata.items():
        sheet.append([key, value])
    
    # Add blank row
    row_num += len(metadata) + 1
    sheet.append([])
    
    # Add results header
    sheet.append(["Index", "Question", "Model Answer", "Actual Answer"])
    
    # Add results
    for row in rows:
        sheet.append([
            row["index"],
            row["question"],
            row["model_answer"],
            row["actual_answer"],
        ])

    workbook.save(output_path)


def get_next_output_path(base_path: Path) -> Path:
    """Generate a unique output path by auto-incrementing the number suffix."""
    if not base_path.exists():
        return base_path
    
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    
    # Extract base name and existing number
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base_name = parts[0]
        counter = int(parts[1]) + 1
    else:
        base_name = stem
        counter = 1
    
    # Find next available number
    while True:
        new_name = f"{base_name}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def load_vector_store_and_agent(
    model: str,
    top_k: int,
    embedding_model: str,
    vector_store_type: str,
    chunk_size: int,
    chunk_overlap: int,
):
    print(f"Loading vector store and retriever (top_k={top_k})...")
    print(f"  Vector Store: {vector_store_type}")
    print(f"  Embedding Model: {embedding_model}")
    if vector_store_type == "faiss":
        print(f"  Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")

    vector_store_path = f"vector_store/{vector_store_type.upper()}/with_chunking"
    vector_store = load_or_create_vector_store(
        vector_store_path=vector_store_path,
        embedding_model=embedding_model,
    )
    retriever_tool, retriever = create_search_retriever(vector_store, top_k=top_k)
    agent = create_legal_agent(retriever_tool, model=model)
    return agent, retriever


def load_qa_split(split: str):
    questions_ds = load_qa_dataset()
    if split not in questions_ds:
        raise ValueError(
            f"Split '{split}' not found in dataset. Available splits: {list(questions_ds.keys())}"
        )
    return questions_ds[split]


def resolve_end_index(total_questions: int, start_index: int, limit: Optional[int]) -> int:
    end_index = total_questions if limit is None else min(start_index + limit, total_questions)
    if start_index >= total_questions:
        raise ValueError(
            f"start_index {start_index} is outside dataset range of 0..{total_questions - 1}"
        )
    return end_index


def evaluate_questions(
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
    agent, _ = load_vector_store_and_agent(
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

    rows = []
    for idx in range(start_index, end_index):
        question = dataset["question"][idx]
        actual = dataset["answer"][idx]

        print(f"[{idx}] Sending question to agent...")
        answer = get_rag_answer(agent, question)

        rows.append(
            {
                "index": idx,
                "question": question,
                "model_answer": answer,
                "actual_answer": actual,
            }
        )

    # Prepare metadata for the Excel file
    metadata = {
        "LLM Model": model,
        "Embedding Model": embedding_model,
        "Vector Store": vector_store_type,
        "Top-K": top_k,
        "Chunk Size": chunk_size,
        "Chunk Overlap": chunk_overlap,
        "Dataset Split": split,
        "Start Index": start_index,
        "Number of Questions": len(rows),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path = get_next_output_path(output_path)
    write_excel(rows, final_output_path, metadata)
    print(f"Saved evaluation results to {final_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a QA dataset with the legal RAG agent and export results to Excel."
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
        default=20,
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
        default=Path("question_answer_results.xlsx"),
        help="Path to the Excel output file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    limit = None if args.limit == 0 else args.limit
    evaluate_questions(
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
