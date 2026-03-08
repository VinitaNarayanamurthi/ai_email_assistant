from typing import Optional

from langchain_openai import ChatOpenAI


def get_chat_model(
    model: str = "gpt-4o",
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    kwargs: dict[str, object] = {
        "model": model,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)  # type: ignore[arg-type]
