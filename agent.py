from langchain.agents import create_agent

from config import LLM_MODEL


def create_legal_agent(retriever_tool, model=None):
    """Create a legal assistant agent that uses the retriever tool."""
    if model is None:
        model = LLM_MODEL

    return create_agent(
        model=model,
        tools=[retriever_tool],
        system_prompt=(
            "You are a helpful legal assistant. For questions regarding legal information, "
            "first call the kb_search tool to retrieve context, then answer succinctly. "
            "You may need to use it multiple times before answering."
            "Answer ONLY using the provided context. If the answer is not present, say I don't know.Do not infer beyond the context."
            "If it is a yes or no question, start with that."
        ),
    )


def get_rag_answer(agent, question):
    """Invoke the agent with a user question and return the final response."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}],
    })
    return result["messages"][-1].content
