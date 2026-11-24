import unittest

from .retrieval_metrics import precision_at_k, recall_at_k


class PrecisionAtKTests(unittest.TestCase):
    def test_precision(self) -> None:
        for case_id, labels, k, expected in (
            ("P01", [1, 1, 1], 3, 1.0),
            ("P02", [0, 0, 0], 3, 0.0),
            ("P03", [1, 0, 1], 3, 2 / 3),
            ("P04", [1, 0, 1], 2, 0.5),
            ("P05", [0, 1, 1], 1, 0.0),
            ("P06", [1], 1, 1.0),
        ):
            with self.subTest(case=case_id):
                result = precision_at_k(labels, k)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected)

    def test_invalid_inputs(self) -> None:
        for labels, k in (
            ([1], 0), ([1], -1), ([1], 2), ([], 1),
            ([1], True), ([1], False), ([1], 1.0), ([1], "1"),
            ([2], 1), ([-1], 1), ([True], 1), ([False], 1),
            ([1.0], 1), (["1"], 1), ([1, 2], 1),
        ):
            with self.subTest(labels=labels, k=k):
                with self.assertRaises(ValueError):
                    precision_at_k(labels, k)

    def test_preserves_rank_order(self) -> None:
        labels = [0, 1, 1]
        self.assertEqual(precision_at_k(labels, 1), 0.0)
        self.assertEqual(labels, [0, 1, 1])


class RecallAtKTests(unittest.TestCase):
    def test_r06_zero_total_is_undefined(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k([0, 0, 0], 3, 0)

    def test_recall(self) -> None:
        for case_id, labels, k, total, expected in (
            ("R01", [1, 1, 0, 1, 1], 5, 10, 0.4),
            ("R02", [1, 1, 1], 3, 3, 1.0),
            ("R03", [1, 0, 0], 3, 5, 0.2),
            ("R04", [0, 0, 0], 3, 5, 0.0),
            ("R05", [1, 0, 1], 2, 4, 0.25),
        ):
            with self.subTest(case=case_id):
                result = recall_at_k(labels, k, total)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, expected)

    def test_invalid_inputs(self) -> None:
        for labels, k, total in (
            ([1], 1, -1), ([0], 1, 0), ([1], 1, True), ([0], 1, False),
            ([1], 1, 1.0), ([1], 1, "1"),
            ([2], 1, 3), ([-1], 1, 3), ([True], 1, 3),
            ([False], 1, 3), ([1.0], 1, 3), (["1"], 1, 3),
            ([1], 0, 1), ([1], -1, 1), ([1], 2, 1), ([], 1, 1),
            ([1], True, 1), ([1], 1.0, 1), ([1], "1", 1),
            ([1, 1, 1], 1, 2), ([1, 2], 1, 3),
        ):
            with self.subTest(labels=labels, k=k, total=total):
                with self.assertRaises(ValueError):
                    recall_at_k(labels, k, total)


if __name__ == "__main__":
    unittest.main()
