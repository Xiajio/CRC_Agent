from langchain_core.tools import tool


@tool
def echo(text: str) -> str:
    """Echo the provided text."""

    return text


@tool
def word_count(text: str) -> int:
    """Count words in the provided text."""

    return len(text.split())


def list_tools():
    """Return the default tool registry."""

    return [echo, word_count]

