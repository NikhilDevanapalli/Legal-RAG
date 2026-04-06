from config import TOP_N_EMBEDDINGS
from data import load_qa_dataset
from vector_store import load_or_create_vector_store
from retriever import create_search_retriever
from agent import create_legal_agent, get_rag_answer


def main():
    vector_store = load_or_create_vector_store()
    retriever_tool, retriever = create_search_retriever(vector_store)
    agent = create_legal_agent(retriever_tool)

    questions_ds = load_qa_dataset()
    sample_question = questions_ds["test"]["question"][11]
    query = sample_question
    print("Query:", query)

    retrieved_docs = retriever.invoke(query)
    print(f"\nRetrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\nDocument {i}:")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Content: {doc.page_content}...")

    answer = get_rag_answer(agent, query)
    print("\nAgent answer:", answer)

    actual_answer = questions_ds["test"]["answer"][11]
    print("\nActual answer:", actual_answer)


if __name__ == "__main__":
    main()
