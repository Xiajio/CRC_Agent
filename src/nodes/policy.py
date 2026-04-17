from typing import List

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..state import AgentState


def build_policy_node(model, tools: List[BaseTool]) -> Runnable:
    """Create a policy node that decides the next action."""

    bound_model = model.bind_tools(tools)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant. "
                    "Answer the user's questions directly. "
                    "Only use tools when specifically needed for the task."
                )
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            SystemMessage(content="Current plan (optional): {plan}"),
        ]
    )
    chain = prompt | bound_model

    def _ensure_message(msg) -> BaseMessage:
        """Make sure we always return a LangChain message object."""

        if isinstance(msg, BaseMessage):
            return msg
        if isinstance(msg, str):
            return AIMessage(content=msg)
        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("text") or str(msg)
            return AIMessage(content=content)
        return AIMessage(content=str(msg))

    def _policy(state: AgentState):
        try:
            response = chain.invoke({"chat_history": state.chat_history, "plan": state.plan or ""})
            response_message = _ensure_message(response)
            # 避免空字符串导致最终输出空白；若包含工具调用则保留 tool_calls 继续走工具节点
            if not getattr(response_message, "content", "").strip():
                if isinstance(response_message, AIMessage) and response_message.tool_calls:
                    response_message = AIMessage(
                        content="(模型请求调用工具)",
                        tool_calls=response_message.tool_calls,
                        additional_kwargs=getattr(response_message, "additional_kwargs", {}),
                    )
                else:
                    response_message = AIMessage(
                        content=str(response) if str(response).strip() else "(LLM 返回空内容)"
                    )
            next_plan = state.plan or "Continue based on latest response."
            # 使用 add_messages reducer，只返回新消息，避免重复
            return {"chat_history": [response_message], "plan": next_plan}
        except Exception as exc:  # noqa: BLE001
            # Print full traceback to help diagnose model/tool binding issues.
            import traceback

            traceback.print_exc()
            return {"error": f"policy_error: {type(exc).__name__}: {exc}"}

    return _policy

