from collections.abc import Sequence
import re
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# Preserve the earlier import location for existing callers.
from .retrieval_metrics import precision_at_k

class TraceResult(TypedDict):
    passed: bool
    reason: str
    tool_requested: bool
    matching_tool_message: bool
    tool_successful: bool
    evidence_received: bool


class EvaluationResult(TypedDict):
    passed: bool
    reason: str


class OutOfScopeResult(EvaluationResult):
    final_answer_available: bool
    unsupported_weather_claim: bool


def evaluate_out_of_scope(messages: Sequence[BaseMessage]) -> OutOfScopeResult:
    """Check controlled weather refusals and explicit unsupported weather claims.

    Temperature units and common condition/forecast statements are recognized.
    These rules have limited English coverage.
    """
    answer = _final_answer(messages).strip().replace("**", "").replace("`", "")
    answer = answer.replace("’", "'")
    temperature = re.search(
        r"(?<!\w)-?\d+(?:\.\d+)?\s*(?:°\s*[CF]?\b|degrees?\b)",
        answer, re.IGNORECASE,
    )
    condition = re.search(
        r"\b(?:London|weather|it)\s+(?:is|will be|looks)\s+"
        r"(?:(?:currently|now|expected to be)\s+)?"
        r"(?:sunny|rainy|cloudy|snowy|raining|snowing|clear|warm|cold)\b"
        r"|\b(?:expect|forecast(?:s|ed)?)\s+(?:rain|snow|sunshine)\b",
        answer, re.IGNORECASE,
    )
    unsupported = bool(temperature or condition)
    boundary = (
        re.search(r"\binvoices?\b", answer, re.IGNORECASE)
        and re.search(r"\bweather\b", answer, re.IGNORECASE)
        and re.search(
            r"\b(?:cannot|can't|unable to)\s+(?:determine|provide|tell|establish|answer)\b"
            r"|\b(?:do not|does not|don't|doesn't)\s+(?:contain|include|provide|establish)\b"
            r"|\b(?:only have|only (?:access|have access) to|limited to|outside (?:my|the) scope)\b",
            answer, re.IGNORECASE,
        )
    )
    if not answer:
        reason = "No nonempty final AIMessage was found."
    elif unsupported:
        reason = "Final answer contains a recognized unsupported weather claim."
    elif not boundary:
        reason = "Final answer does not explain the invoice evidence boundary for weather."
    else:
        reason = "Final answer respects the invoice evidence boundary for weather."
    return {
        "passed": bool(answer and boundary and not unsupported),
        "reason": reason,
        "final_answer_available": bool(answer),
        "unsupported_weather_claim": unsupported,
    }


def _final_answer(messages: Sequence[BaseMessage]) -> str:
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return messages[-1].text
    return ""


def evaluate_abstention(messages: Sequence[BaseMessage]) -> EvaluationResult:
    """Check controlled payment-status phrases, with limited English coverage.

    Claims such as "Payment went through successfully" are outside the recognized
    assertion grammar. A passing result cannot establish arbitrary answer grounding.
    """
    answer = _final_answer(messages).strip().replace("**", "").replace("`", "")
    answer = answer.replace("’", "'")
    if not answer:
        return {"passed": False, "reason": "No nonempty final AIMessage was found."}

    uncertainty = (
        r"\b(?:cannot|can't|unable to)\s+(?:determine|confirm|establish|verify|tell)\b"
        r"|\b(?:does not|doesn't)\s+(?:establish|confirm|indicate|show|provide)\b"
        r"|\b(?:insufficient|not enough|no)\s+(?:information|evidence)\b"
        r"|\b(?:payment status|status of (?:the )?payment)\s+"
        r"(?:(?:is|was|remains)\s+)?(?:unknown|unavailable|unclear|not available|not provided)\b"
        r"|\b(?:not possible to|not able to)\s+(?:determine|confirm|verify)\b"
    )
    assertion = (
        r"\b(?:invoice(?:\s+(?:#?\d+|number\s+\d+))?|it|this|payment status)"
        r"\s+(?:is|was|has been|had been)\s+(?:definitely\s+|fully\s+)?(?:not\s+)?(?:paid|unpaid)\b"
        r"|\bpayment\s+(?:is|was|has been)\s+(?:not\s+)?(?:completed|received|confirmed)\b"
    )
    for clause in re.split(r"[.!?;\n]|\b(?:but|however|yet)\b", answer, flags=re.IGNORECASE):
        for match in re.finditer(assertion, clause, re.IGNORECASE):
            prefix = clause[:match.start()]
            # An assertion inside "cannot determine whether ..." is an open question.
            uncertain_scope = re.search(
                r"\b(?:(?:cannot|can't|unable to)\s+(?:determine|confirm|establish|verify|tell)"
                r"|no\s+(?:information|evidence)\s+(?:on|about|as to)"
                r"|(?:does not|doesn't)\s+(?:include|provide)\s+information\s+(?:regarding|about)"
                r"|(?:does not|doesn't)\s+(?:confirm|establish|show)(?=\s+(?:whether|if)\b))"
                r"\s+(?:whether|if|that)\s+(?:(?:the|this)\s+)?$",
                prefix, re.IGNORECASE,
            )
            if not uncertain_scope:
                return {"passed": False, "reason": "Final answer asserts an unsupported payment status."}

    if not re.search(r"\b(?:paid|unpaid|payment)\b", answer, re.IGNORECASE) or not re.search(
        uncertainty, answer, re.IGNORECASE
    ):
        return {"passed": False, "reason": "Final answer does not clearly communicate insufficient payment evidence."}
    return {"passed": True, "reason": "Final answer acknowledges insufficient payment evidence without a recognized definite status claim."}


def evaluate_answer_correctness(
    messages: Sequence[BaseMessage], expected_amount: str
) -> EvaluationResult:
    """Check a limited T01 answer format.

    The same sentence must contain the USD amount, membership, and charge/charged.
    Equivalent wording such as "membership cost" can produce a false negative.
    """
    answer = _final_answer(messages).replace("**", "").replace("`", "")
    amount_pattern = rf"\bUSD\s*\$?\s*{re.escape(expected_amount)}(?!\d|[.,]\d)"
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if (
            re.search(amount_pattern, sentence, re.IGNORECASE)
            and re.search(r"\bmembership\b", sentence, re.IGNORECASE)
            and re.search(r"\bcharg(?:e|ed)\b", sentence, re.IGNORECASE)
        ):
            return {"passed": True, "reason": "Final answer states the expected USD membership charge."}
    return {"passed": False, "reason": "Final answer does not state the expected USD membership charge."}


def evaluate_invoice_number(
    messages: Sequence[BaseMessage], expected_invoice_number: str
) -> EvaluationResult:
    """Check the controlled T02 format for an explicitly identified invoice number."""
    answer = _final_answer(messages).replace("**", "").replace("`", "")
    pattern = (
        r"\binvoice\s+(?:number\s*(?:(?:is|was)\s*|[:#]\s*)?|#\s*)#?\s*"
        rf"{re.escape(expected_invoice_number)}\b"
    )
    if re.search(pattern, answer, re.IGNORECASE):
        return {"passed": True, "reason": "Final answer identifies the expected invoice number."}
    return {"passed": False, "reason": "Final answer does not identify the expected invoice number."}


def evaluate_citation(
    messages: Sequence[BaseMessage],
    expected_tool: str,
    expected_source: str,
    expected_evidence: str,
    expected_description: str,
) -> EvaluationResult:
    """Match a final source-page citation to a successful retrieved evidence block.

    Source fields use filenames in the current stored records and tool output.
    """
    answer = _final_answer(messages)
    # Accept the prototype's source.pdf, page N citations, including Markdown emphasis.
    citation_pattern = rf"{re.escape(expected_source)}[\s,*\x60)\]]*\(?pages?\s+(\d+)\b"
    cited_pages = {
        int(match) for match in re.findall(citation_pattern, answer, re.IGNORECASE)
    }
    if not cited_pages:
        return {"passed": False, "reason": "Final answer has no recognized source and page citation."}

    pending_ids: set[str] = set()
    for message in messages[:-1]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                query = call["args"].get("query")
                if call["name"] == expected_tool and isinstance(query, str) and query.strip() and call.get("id"):
                    pending_ids.add(call["id"])
        if not isinstance(message, ToolMessage) or message.tool_call_id not in pending_ids:
            continue
        pending_ids.remove(message.tool_call_id)
        if getattr(message, "status", "success") != "success":
            continue
        for block in re.split(r"(?m)^Result \d+\s*\n", message.text):
            source = re.search(r"(?m)^Source: (.+)$", block)
            page = re.search(r"(?m)^Page: (\d+)\s*$", block)
            _, separator, evidence = block.partition("\nEvidence:\n")
            if (
                source and source.group(1).strip() == expected_source
                and page and int(page.group(1)) in cited_pages
                and separator and re.search(rf"{re.escape(expected_evidence)}(?!\d)", evidence)
                and expected_description in evidence
            ):
                return {
                    "passed": True,
                    "reason": f"Cited {expected_source}, page {page.group(1)} contains the expected evidence.",
                }
    return {"passed": False, "reason": "No cited source page has matching retrieved evidence."}


def evaluate_tool_trace(
    messages: Sequence[BaseMessage],
    expected_tool: str,
    expected_evidence: str,
) -> TraceResult:
    """Check that the expected tool executed and returned evidence."""
    if not expected_tool.strip() or not expected_evidence.strip():
        raise ValueError("Expected tool and evidence must contain text.")

    pending_ids: set[str] = set()
    result: TraceResult = {
        "passed": False,
        "reason": "No expected tool request with a valid query and call ID was found.",
        "tool_requested": False,
        "matching_tool_message": False,
        "tool_successful": False,
        "evidence_received": False,
    }
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                query = call["args"].get("query")
                call_id = call.get("id")
                if (
                    call["name"] == expected_tool
                    and isinstance(query, str) and query.strip()
                    and isinstance(call_id, str) and call_id.strip()
                ):
                    pending_ids.add(call_id)
                    result["tool_requested"] = True
                    result["reason"] = "No ToolMessage matched a preceding valid request ID."

        if isinstance(message, ToolMessage) and message.tool_call_id in pending_ids:
            pending_ids.remove(message.tool_call_id)
            result["matching_tool_message"] = True
            if getattr(message, "status", "success") != "success":
                result["reason"] = "Matching tool result reported an error."
                continue
            result["tool_successful"] = True
            if expected_evidence not in message.text:
                result["reason"] = "Successful tool result lacked the expected evidence."
                continue
            result["evidence_received"] = True
            result["passed"] = True
            result["reason"] = "Matching tool call and evidence found."
            return result
    return result
