import unittest

from langchain_core.messages import AIMessage, ToolMessage

from .evaluators import evaluate_abstention, evaluate_answer_correctness, evaluate_citation, evaluate_tool_trace
from .test_cases import T01


class ToolTraceTests(unittest.TestCase):
    def test_synthetic_histories(self) -> None:
        request = AIMessage(content="", tool_calls=[{
            "name": T01["expected_tool"], "args": {"query": "monthly membership"}, "id": "call-a",
        }])
        evidence = ToolMessage(content=T01["expected_evidence"], tool_call_id="call-a")
        blank_query = AIMessage(content="", tool_calls=[{
            "name": T01["expected_tool"], "args": {"query": " "}, "id": "call-a",
        }])
        cases = [
            ("E01", [request, evidence], True),
            ("E02", [evidence], False),
            ("E03", [request, ToolMessage(content=evidence.content, tool_call_id="call-b")], False),
            ("E04", [request, ToolMessage(content=evidence.content, tool_call_id="call-a", status="error")], False),
            ("E05", [request, ToolMessage(content="No invoice evidence found.", tool_call_id="call-a")], False),
            ("blank query", [blank_query, evidence], False),
            ("out of order", [evidence, request], False),
        ]
        for case_id, messages, expected in cases:
            with self.subTest(case=case_id):
                result = evaluate_tool_trace(messages, T01["expected_tool"], T01["expected_evidence"])
                self.assertEqual(result["passed"], expected)


class AnswerAndCitationTests(unittest.TestCase):
    def test_correctness(self) -> None:
        for answer, expected in (
            ("The monthly membership charge was USD $37.50.", True),
            ("The membership was charged **USD $37.50** (sample.pdf, page 1).", True),
            ("The membership charge was USD $50.00.", False),
            ("The membership charge was USD $37.500.", False),
            ("USD $37.50", False),
            ("The membership charge was EUR 37.50.", False),
            # Documents the controlled rule's known false negative.
            ("The monthly membership cost is USD $37.50.", False),
            ("", False),
        ):
            with self.subTest(answer=answer):
                result = evaluate_answer_correctness([AIMessage(content=answer)], T01["expected_amount"])
                self.assertEqual(result["passed"], expected)

    def test_no_final_answer(self) -> None:
        request = AIMessage(content="", tool_calls=[{
            "name": "search_invoices", "args": {"query": "monthly membership"}, "id": "call-a",
        }])
        messages = [request, ToolMessage(
            content=(
                "Result 1\nSource: sample.pdf\nPage: 1\n\nEvidence:\n"
                "Premium Membership - Monthly\nUSD $37.50"
            ),
            tool_call_id="call-a",
        )]
        trace = evaluate_tool_trace(messages, T01["expected_tool"], T01["expected_evidence"])
        answer = evaluate_answer_correctness(messages, T01["expected_amount"])
        citation = evaluate_citation(
            messages, T01["expected_tool"], T01["expected_source"],
            T01["expected_evidence"], T01["expected_description"],
        )
        self.assertTrue(trace["passed"])
        self.assertFalse(answer["passed"])
        self.assertFalse(citation["passed"])

    def test_citation(self) -> None:
        request = AIMessage(content="", tool_calls=[{
            "name": "search_invoices", "args": {"query": "monthly membership"}, "id": "call-a",
        }])
        evidence = (
            "Result 1\nSource: sample.pdf\nPage: 1\nChunk ID: sample-p0-c1\n"
            "Distance: 0.8\n\nEvidence:\nPremium Membership - Monthly\nUSD $37.50"
        )
        answer = "The membership charge was USD $37.50 (sample.pdf, page 1)."
        cases = [
            ("supported", evidence, answer, "call-a", "success", True),
            ("bold citation", evidence, answer.replace("sample.pdf", "**sample.pdf**"), "call-a", "success", True),
            ("wrong page", evidence, answer.replace("page 1", "page 2"), "call-a", "success", False),
            ("wrong source", evidence, answer.replace("sample.pdf", "other.pdf"), "call-a", "success", False),
            ("no citation", evidence, "The membership charge was USD $37.50.", "call-a", "success", False),
            ("wrong amount", evidence.replace("USD $37.50", "USD $50.00"), answer, "call-a", "success", False),
            ("wrong description", evidence.replace("Premium Membership - Monthly", "Other service"), answer, "call-a", "success", False),
            ("unrelated result", evidence, answer, "call-b", "success", False),
            ("failed result", evidence, answer, "call-a", "error", False),
            (
                "evidence on another page",
                evidence.replace("USD $37.50", "USD $50.00")
                + "\n\n" + evidence.replace("Result 1", "Result 2").replace("Page: 1", "Page: 2"),
                answer, "call-a", "success", False,
            ),
        ]
        for name, text, final, call_id, status, expected in cases:
            with self.subTest(case=name):
                messages = [
                    request, ToolMessage(content=text, tool_call_id=call_id, status=status),
                    AIMessage(content=final),
                ]
                result = evaluate_citation(
                    messages, T01["expected_tool"], T01["expected_source"],
                    T01["expected_evidence"], T01["expected_description"],
                )
                self.assertEqual(result["passed"], expected, result["reason"])


class AbstentionTests(unittest.TestCase):
    def test_payment_status_answers(self) -> None:
        cases = [
            ("A01", "The available text does not establish payment status.", True),
            ("N01", "Invoice 24680135 was paid.", False),
            ("N02", "I cannot determine whether invoice 24680135 was paid.", True),
            ("N03", "I cannot determine whether the invoice 24680135 was paid.", True),
            ("N04", "I cannot determine whether invoice #24680135 was paid.", True),
            ("N05", "I cannot determine whether invoice 24680135 was paid, but it was paid.", False),
            ("R01", "The extracted evidence does not confirm whether it was paid.", True),
            ("R02", "There is no information on whether this invoice was paid.", True),
            ("R03", "The invoice was paid.", False),
            ("R04", "I cannot confirm whether it was paid, but it was paid.", False),
            ("different invoice number", "Invoice 24680135 was paid.", False),
            ("different invoice contradiction", "Payment status is unknown. Invoice 24680135 was paid.", False),
            ("different invoice uncertainty", "I cannot determine whether invoice 24680135 was paid.", True),
            (
                "provide information live answer",
                "The retrieved evidence for invoice #13594027 indicates a charge of USD $37.50 for a Premium Membership, "
                "but it does not provide information about whether this invoice was paid or if there was a payment status available. "
                "Since payment status details are absent in the extracted evidence, I cannot confirm the payment for invoice #13594027. "
                "The relevant documents are found on page 1, page 2, and page 4 of the source (sample.pdf).",
                True,
            ),
            ("provide information contradiction", "The evidence does not provide information about whether this invoice was paid, but invoice 24680135 was paid.", False),
            ("A02", "I cannot determine whether the invoice was paid.", True),
            ("A03", "The invoice was paid.", False),
            ("A04", "The invoice was unpaid.", False),
            ("A05", "I cannot confirm payment, but the invoice was paid.", False),
            ("A06", "", False),
            ("charge", "The amount charged was USD $37.50, but payment status is unknown.", True),
            ("not paid", "It was not paid.", False),
            ("contradiction with ID", "I cannot confirm payment status, but invoice 13594027 was paid.", False),
            ("unknown", "Payment status is unavailable in the extracted text.", True),
            ("uncertain pronoun", "I cannot determine whether it was paid.", True),
            ("uncertain invoice number", "I cannot determine whether invoice 13594027 was paid.", True),
            ("negated confirmation", "The extracted evidence does not confirm whether it was paid.", True),
            ("contraction confirmation", "The extracted evidence doesn't establish if the invoice was paid.", True),
            ("negated showing", "The text does not show whether invoice 13594027 was unpaid.", True),
            (
                "new captured live answer",
                "The retrieved evidence for invoice 13594027 does not include information regarding whether the invoice was paid. "
                "It shows details about the charge, including a total amount and discounts, but does not confirm payment status. \n\n"
                "The relevant details from the evidence state:\n"
                "- Total amount: USD $37.50\n"
                "- Description: Premium Membership - Monthly\n"
                "- Date: 06/23/2018\n\n"
                "Unfortunately, without explicit payment status provided in the documents (e.g., a stamp indicating payment), "
                "I cannot determine if the invoice was paid. The evidence comes from sample.pdf, pages 1, 2, and 4.",
                True,
            ),
            ("missing information contradiction", "The evidence does not include information regarding whether the invoice was paid. The invoice was paid.", False),
            ("scoped contradiction", "I cannot confirm whether it was paid, but it was paid.", False),
            ("negated confirmation contradiction", "The evidence does not confirm whether it was paid, but it was paid.", False),
            ("unknown then paid", "Payment status is unknown. The invoice was paid.", False),
            (
                "captured live answer",
                "The retrieved documents for invoice #13594027 provide details about the charges, "
                "which include a total of USD $37.50 after a discount for a Premium Membership. "
                "However, there is no information on whether this invoice was paid. "
                "The visual payment status stamps present in the PDF were not captured during text extraction. "
                "Therefore, I cannot confirm the payment status based on the provided evidence (Source: sample.pdf, Page 2).",
                True,
            ),
            ("live uncertainty wording", "There is no information on whether this invoice was paid. I cannot confirm the payment status based on the provided evidence.", True),
            ("uncertainty followed by claim", "There is no information on whether this invoice was paid. However, the invoice was paid.", False),
            ("whitespace", "   ", False),
            ("irrelevant", "I cannot determine the weather.", False),
            ("contradiction", "Payment status is unknown. The invoice was unpaid.", False),
        ]
        for name, answer, expected in cases:
            with self.subTest(case=name):
                result = evaluate_abstention([AIMessage(content=answer)])
                self.assertEqual(result["passed"], expected, result["reason"])
        self.assertFalse(evaluate_abstention([])["passed"])
        self.assertFalse(evaluate_abstention([ToolMessage(content="Payment status unknown.", tool_call_id="a")])["passed"])


if __name__ == "__main__":
    unittest.main()
