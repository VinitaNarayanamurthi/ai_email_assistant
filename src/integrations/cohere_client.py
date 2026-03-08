from typing import Optional


def call_cohere(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
) -> str:
    try:
        from litellm import completion  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("litellm is required for Cohere integration. Run: pip install litellm") from e

    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = completion(**kwargs)  # type: ignore[arg-type]
    return str(response.choices[0].message.content)  # type: ignore[attr-defined]
