import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openrouter import ChatOpenRouter


def initialize_model() -> ChatOpenRouter:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set LLM_API_KEY in the project .env file or process environment.")

    timeout_seconds = int(os.environ["LLM_TIMEOUT_SECONDS"])
    output_tokens = int(os.environ["LLM_OUTPUT_TOKENS"])
    if timeout_seconds <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive integer.")
    if output_tokens <= 0:
        raise ValueError("LLM_OUTPUT_TOKENS must be a positive integer.")

    return ChatOpenRouter(
        model=os.environ["LLM_MODEL"],
        api_key=api_key,
        base_url=os.environ["LLM_BASE_URL"],
        timeout=timeout_seconds * 1000,
        max_tokens=output_tokens,
        max_retries=int(os.environ["LLM_RETRY_ATTEMPTS"]),
    )


def main() -> None:
    try:
        model = initialize_model()
    except (KeyError, ValueError) as error:
        raise SystemExit(f"Configuration error: {error}") from error

    response = model.invoke("Explain what an invoice is in one sentence.")
    if not isinstance(response, AIMessage) or not response.text.strip():
        raise ValueError("Expected an AIMessage with a nonempty text response.")
    print("Provider: OpenRouter")
    print(f"Model: {model.model}")
    print(f"Response type: {type(response).__name__}")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    main()
