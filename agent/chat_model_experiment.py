import argparse

from httpx import ConnectError
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError


DEFAULT_MODEL = "qwen3:0.6b"


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a local Ollama chat model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    model = ChatOllama(
        model=args.model,
        temperature=0,
        reasoning=False,
        client_kwargs={"timeout": 180},
    )
    try:
        response = model.invoke("Say hello in one short sentence.")
    except (ConnectionError, ConnectError) as error:
        raise SystemExit("Cannot connect to Ollama. Start Ollama and retry.") from error
    except ResponseError as error:
        raise SystemExit(f"Ollama error: {error.error}") from error
    if not isinstance(response, AIMessage) or not response.text.strip():
        raise ValueError("Expected an AIMessage containing a nonempty text response.")

    print(f"Message type: {type(response).__name__}")
    print(f"Content: {response.text}")
    print(f"Tool calls: {response.tool_calls}")
    print(f"Response metadata: {response.response_metadata}")
    print(f"Usage metadata: {response.usage_metadata}")


if __name__ == "__main__":
    main()
