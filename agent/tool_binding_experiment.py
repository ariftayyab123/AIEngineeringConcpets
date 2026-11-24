from langchain_core.messages import AIMessage

from agent.model_experiment import initialize_model
from agent_tools.invoice_search_tool import search_invoices


def main() -> None:
    model = initialize_model()
    model_with_tools = model.bind_tools([search_invoices])
    response = model_with_tools.invoke(
        "Use search_invoices to find the monthly membership charge."
    )
    if not isinstance(response, AIMessage):
        raise ValueError("Expected an AIMessage.")

    print(f"Response type: {type(response).__name__}")
    print(f"Tool calls: {response.tool_calls}")
    if response.invalid_tool_calls:
        raise ValueError(f"Model returned invalid tool calls: {response.invalid_tool_calls}")
    if not response.tool_calls:
        raise ValueError("Model did not request a tool call.")

    for call in response.tool_calls:
        if call["name"] != "search_invoices":
            raise ValueError(f"Unexpected tool name: {call['name']}")
        query = call["args"].get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Tool call must contain a nonempty string query.")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Tool call must contain a nonempty call ID.")

    print("Structured tool-call validation: PASSED")


if __name__ == "__main__":
    main()
