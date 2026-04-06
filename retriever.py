from langchain_core.tools import create_retriever_tool

from config import TOP_N_EMBEDDINGS


def create_search_retriever(vector_store, top_k=TOP_N_EMBEDDINGS):
    """Create a retriever and tool for the legal knowledge base."""
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return create_retriever_tool(
        retriever,
        name="kb_search",
        description="Search the legal knowledge base for information.",
    )
