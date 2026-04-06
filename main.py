from config import TOP_N_EMBEDDINGS
from data import load_qa_dataset
from vector_store import load_or_create_vector_store
from retriever import create_search_retriever
from agent import create_legal_agent, get_rag_answer


def main():
    vector_store = load_or_create_vector_store()
    retriever_tool = create_search_retriever(vector_store)
    agent = create_legal_agent(retriever_tool)

    questions_ds = load_qa_dataset()
    sample_question = questions_ds["test"]["question"][11]
    print("Sample question:", sample_question)

    answer = get_rag_answer(
        agent,
        "Can the defence use good character evidence to argue the accused is innocent?",
    )
    print("Agent answer:", answer)


if __name__ == "__main__":
    main()
