import unittest

from langchain_core.messages import AIMessage, ToolMessage

from .evaluators import evaluate_citation, evaluate_invoice_number, evaluate_tool_trace
from .test_cases import T02


class InvoiceNumberTests(unittest.TestCase):
    def test_final_answer(self) -> None:
        for answer, expected in (
            ("The invoice number is **13594027** (sample.pdf, page 1).", True),
            ("Invoice #13594027", True),
            ("Invoice number: 13594027", True),
            ("The invoice number is #13594027, as found on multiple pages of the document (sample.pdf, pages 1-3).", True),
            ("The invoice number is 135940270.", False),
            ("The invoice number is 24680135.", False),
            ("Transaction ID: 13594027", False),
            ("", False),
        ):
            with self.subTest(answer=answer):
                result = evaluate_invoice_number([AIMessage(content=answer)], T02["expected_invoice_number"])
                self.assertEqual(result["passed"], expected)
        self.assertFalse(evaluate_invoice_number([], T02["expected_invoice_number"])["passed"])

    def test_retrieved_support(self) -> None:
        request = AIMessage(content="", tool_calls=[{
            "name": "search_invoices", "args": {"query": "invoice number"}, "id": "call-t02",
        }])
        for page, evidence, call_id, status, expected in (
            (1, "Invoice #13594027", "call-t02", "success", True),
            (2, "Invoice #13594027", "call-t02", "success", False),
            (1, "Invoice #24680135", "call-t02", "success", False),
            (1, "Invoice #135940270", "call-t02", "success", False),
            (1, "Invoice #13594027", "unrelated", "success", False),
            (1, "Invoice #13594027", "call-t02", "error", False),
        ):
            with self.subTest(page=page, evidence=evidence, call_id=call_id, status=status):
                messages = [
                    request,
                    ToolMessage(
                        content=f"Result 1\nSource: sample.pdf\nPage: {page}\n\nEvidence:\n{evidence}",
                        tool_call_id=call_id, status=status,
                    ),
                    AIMessage(content="The invoice number is 13594027 (sample.pdf, page 1)."),
                ]
                citation = evaluate_citation(
                    messages, T02["expected_tool"], T02["expected_source"],
                    T02["expected_evidence"], T02["expected_description"],
                )
                self.assertEqual(citation["passed"], expected)
        trace = evaluate_tool_trace(messages, T02["expected_tool"], T02["expected_evidence"])
        self.assertFalse(trace["passed"])

    def test_captured_answer_citation(self) -> None:
        messages = [
            AIMessage(content="", tool_calls=[{
                "name": "search_invoices", "args": {"query": "invoice number"}, "id": "call-t02",
            }]),
            ToolMessage(
                content="Result 1\nSource: sample.pdf\nPage: 1\n\nEvidence:\nInvoice #13594027",
                tool_call_id="call-t02",
            ),
            AIMessage(content="The invoice number is #13594027, as found on multiple pages of the document (sample.pdf, pages 1-3)."),
        ]
        result = evaluate_citation(
            messages, T02["expected_tool"], T02["expected_source"],
            T02["expected_evidence"], T02["expected_description"],
        )
        self.assertTrue(result["passed"], result["reason"])

    def test_exact_filename_citation_regression(self) -> None:
        evidence = [
            AIMessage(content="", tool_calls=[{
                "name": "search_invoices", "args": {"query": "invoice number"}, "id": "call-t02",
            }]),
            ToolMessage(
                content="Result 1\nSource: sample.pdf\nPage: 2\n\nEvidence:\nInvoice #13594027",
                tool_call_id="call-t02",
            ),
        ]
        for answer, expected in (
            (
                "The invoice number is **#13594027**. This information is found on "
                "page 1, page 2, and page 4 of the sample PDF.",
                False,
            ),
            ("The invoice number is **#13594027** (sample.pdf, page 2).", True),
            ("The invoice number is **#13594027** (sample.pdf, page 1).", False),
        ):
            with self.subTest(answer=answer):
                result = evaluate_citation(
                    [*evidence, AIMessage(content=answer)],
                    T02["expected_tool"], T02["expected_source"],
                    T02["expected_evidence"], T02["expected_description"],
                )
                self.assertEqual(result["passed"], expected, result["reason"])


if __name__ == "__main__":
    unittest.main()
