import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import VECTOR_STORE_PATH, EMBEDDING_MODEL
from data import load_corpus


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


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


def load_or_create_vector_store():
    """Load an existing FAISS index or create it from the corpus."""
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_STORE_PATH):
        print("Loading existing vector store...")
        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Creating new vector store...")
    documents = load_corpus()
    print("Total documents:", len(documents))
    return build_vector_store(documents, embeddings)
