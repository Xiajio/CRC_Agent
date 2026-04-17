from langchain_core.messages import AIMessage

from ..state import AgentState


def handle_error(state: AgentState):
    """Record the error and emit a graceful message."""

    message = state.error or "Unknown error."
    return {"chat_history": [AIMessage(content=f"Recovered from: {message}")], "error": None}

