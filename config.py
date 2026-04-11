import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
VECTOR_STORE_PATH = "vector_store/FAISS/with_chunking"
TOP_N_EMBEDDINGS = 5
DATASET_NAME = "isaacus/legal-rag-bench"
CORPUS_SPLIT = "test"
QA_SUBSET = "qa"
