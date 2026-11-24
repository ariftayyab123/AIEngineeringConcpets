import unittest

from langchain_core.messages import AIMessage, ToolMessage

from .evaluators import evaluate_out_of_scope


class OutOfScopeTests(unittest.TestCase):
    def test_weather_answers(self) -> None:
        cases = [
            ("O01", "The invoice documents do not contain weather information.", True),
            ("O02", "I cannot determine London's weather from the available invoices.", True),
            ("O03", "London is currently 18°C and sunny.", False),
            ("O04", "I only have invoice data, but London is 18°C.", False),
            ("O05", "The weather is USD $37.50.", False),
            ("O06", "", False),
            ("whitespace", "   ", False),
            ("contradictory conditions", "The invoice documents do not contain weather information, but London is sunny.", False),
            ("contradictory forecast", "I cannot determine weather from invoices. Expect rain.", False),
            ("invoice mention only", "Invoice total: USD $37.50.", False),
            (
                "generic live refusal lacks boundary",
                "I'm not able to provide current weather information. You may want to check a reliable "
                "weather website or app for the latest updates on London's weather.",
                False,
            ),
        ]
        for name, answer, expected in cases:
            with self.subTest(case=name):
                result = evaluate_out_of_scope([AIMessage(content=answer)])
                self.assertEqual(result["passed"], expected, result["reason"])

    def test_optional_tool_execution(self) -> None:
        final = AIMessage(content="The invoice documents do not contain weather information.")
        self.assertTrue(evaluate_out_of_scope([final])["passed"])
        self.assertTrue(evaluate_out_of_scope([
            ToolMessage(content="Irrelevant invoice evidence.", tool_call_id="call-weather"), final,
        ])["passed"])

    def test_missing_final_answer(self) -> None:
        for messages in ([], [ToolMessage(content="Invoice weather", tool_call_id="call-weather")]):
            with self.subTest(messages=messages):
                result = evaluate_out_of_scope(messages)
                self.assertFalse(result["passed"])
                self.assertFalse(result["final_answer_available"])


if __name__ == "__main__":
    unittest.main()
