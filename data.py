from datasets import load_dataset
from langchain_core.documents import Document

from config import DATASET_NAME, CORPUS_SPLIT, QA_SUBSET


def load_corpus():
    """Load the corpus dataset and return a list of Document objects."""
    ds = load_dataset(DATASET_NAME, "corpus")
    data = ds[CORPUS_SPLIT]

    documents = []
    for row in data:
        documents.append(
            Document(
                page_content=row["text"],
                metadata={
                    "id": row["id"],
                    "title": row["title"],
                    "footnotes": row["footnotes"],
                },
            )
        )

    return documents


def load_qa_dataset():
    """Load the QA dataset for evaluation or examples."""
    return load_dataset(DATASET_NAME, QA_SUBSET)
