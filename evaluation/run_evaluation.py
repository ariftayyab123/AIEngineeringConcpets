import os
import argparse

from langchain_core.messages import HumanMessage, ToolMessage

from agent.invoice_agent import build_invoice_agent

from .evaluators import evaluate_abstention, evaluate_answer_correctness, evaluate_citation, evaluate_invoice_number, evaluate_out_of_scope, evaluate_tool_trace
from .test_cases import T01, T02, T05, T06


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one invoice evaluation case.")
    parser.add_argument("--test", choices=("T01", "T02", "T05", "T06"), default="T01")
    parser.add_argument("--debug", action="store_true", help="Print the final agent message.")
    args = parser.parse_args()
    test_case = {"T01": T01, "T02": T02, "T05": T05, "T06": T06}[args.test]
    agent = build_invoice_agent()
    result = agent.invoke({"messages": [HumanMessage(content=test_case["question"])]})
    if args.debug:
        final_message = result["messages"][-1]
        print("Final message type:", type(final_message).__name__)
        print("Final answer:", final_message.text)
    if test_case["id"] == "T06":
        boundary = evaluate_out_of_scope(result["messages"])
        tool_errors = sum(
            isinstance(message, ToolMessage) and message.status == "error"
            for message in result["messages"]
        )
        print(f"Model: {os.environ['LLM_MODEL']}")
        print(f"Test ID: {test_case['id']}")
        print(f"Question category: {test_case['category']}")
        print(f"Final answer available: {'YES' if boundary['final_answer_available'] else 'NO'}")
        print(f"Unsupported weather claim: {'YES' if boundary['unsupported_weather_claim'] else 'NO'}")
        print(f"Tool errors: {tool_errors}")
        print(f"Out-of-scope evaluation: {'PASS' if boundary['passed'] else 'FAIL'}")
        print(f"Reason: {boundary['reason']}")
        print(f"Overall: {'PASS' if boundary['passed'] else 'FAIL'}")
        if not boundary["passed"]:
            raise SystemExit(1)
        return
    trace = evaluate_tool_trace(result["messages"], test_case["expected_tool"], test_case["expected_evidence"])
    if test_case["id"] == "T05":
        abstention = evaluate_abstention(result["messages"])
        passed = trace["passed"] and abstention["passed"]
        print(f"Model: {os.environ['LLM_MODEL']}")
        print(f"Test ID: {test_case['id']}")
        print(f"Tool trace: {'PASS' if trace['passed'] else 'FAIL'}")
        print(f"Reason: {trace['reason']}")
        print(f"Abstention: {'PASS' if abstention['passed'] else 'FAIL'}")
        print(f"Reason: {abstention['reason']}")
        print(f"Overall: {'PASS' if passed else 'FAIL'}")
        if not passed:
            raise SystemExit(1)
        return
    if test_case["id"] == "T02":
        correctness = evaluate_invoice_number(result["messages"], T02["expected_invoice_number"])
    else:
        correctness = evaluate_answer_correctness(result["messages"], T01["expected_amount"])
    citation = evaluate_citation(
        result["messages"], test_case["expected_tool"], test_case["expected_source"],
        test_case["expected_evidence"], test_case["expected_description"],
    )

    print(f"Model: {os.environ['LLM_MODEL']}")
    print(f"Test ID: {test_case['id']}")
    print(f"Expected tool: {test_case['expected_tool']}")
    print()
    print(f"Tool request found: {'YES' if trace['tool_requested'] else 'NO'}")
    print(f"Matching ToolMessage: {'YES' if trace['matching_tool_message'] else 'NO'}")
    print(f"Tool execution successful: {'YES' if trace['tool_successful'] else 'NO'}")
    print(f"Expected evidence found: {'YES' if trace['evidence_received'] else 'NO'}")
    print()
    print(f"Trace evaluation: {'PASS' if trace['passed'] else 'FAIL'}")
    print(f"Reason: {trace['reason']}")
    print(f"Answer correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
    print(f"Reason: {correctness['reason']}")
    print(f"Citation validation: {'PASS' if citation['passed'] else 'FAIL'}")
    print(f"Reason: {citation['reason']}")
    if not all((trace["passed"], correctness["passed"], citation["passed"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
