import os
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import VECTOR_STORE_PATH, EMBEDDING_MODEL
from data import load_corpus


def get_embeddings(embedding_model=None):
    if embedding_model is None:
        embedding_model = EMBEDDING_MODEL
    return OpenAIEmbeddings(model=embedding_model)


def build_vector_store(documents, embeddings):
    """Chunk the documents and create a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print("Total chunks:", len(chunked_documents))

    vector_store = FAISS.from_documents(
        chunked_documents,
        embedding=embeddings,
    )
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store saved to {VECTOR_STORE_PATH}")
    return vector_store


def load_or_create_vector_store(vector_store_path=None, embedding_model=None):
    """Load an existing FAISS index or create it from the corpus.
    
    Args:
        vector_store_path: Path to the vector store. Defaults to VECTOR_STORE_PATH from config.
        embedding_model: Embedding model to use. Defaults to EMBEDDING_MODEL from config.
    """
    if vector_store_path is None:
        vector_store_path = VECTOR_STORE_PATH
    
    embeddings = get_embeddings(embedding_model=embedding_model)

    if os.path.exists(vector_store_path):
        print("Loading existing vector store...")
        return FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Creating new vector store...")
    documents = load_corpus()
    print("Total documents:", len(documents))
    return build_vector_store(documents, embeddings)


def load_or_create_chroma_vector_store():
    """Alias for compatibility with existing entrypoints."""
    return load_or_create_vector_store()
