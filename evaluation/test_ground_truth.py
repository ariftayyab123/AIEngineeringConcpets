from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

from . import ground_truth_experiment as experiment
from .ground_truth import REVIEWED_CHUNKS


class GroundTruthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = list(REVIEWED_CHUNKS)
        self.records = {
            "ids": self.ids,
            "documents": [REVIEWED_CHUNKS[key]["text"] for key in self.ids],
            "metadatas": [
                {"source": REVIEWED_CHUNKS[key]["source"], "page": REVIEWED_CHUNKS[key]["page"]}
                for key in self.ids
            ],
        }
        self.collection = Mock()
        self.collection.get.return_value = self.records

    def test_ranked_id_mapping(self) -> None:
        # Mixed synthetic labels expose accidental reliance on text or record order.
        labels = {key: int(index < 6) for index, key in enumerate(self.ids)}
        ranked_ids = [self.ids[6], self.ids[0], self.ids[1]]
        chunks = [
            {"id": key, **REVIEWED_CHUNKS[key], "distance": 1.0} for key in ranked_ids
        ]
        with (
            patch.object(experiment, "open_invoice_collection", return_value=self.collection),
            patch.object(experiment, "search_invoice_chunks", return_value=chunks),
            patch.object(experiment, "GROUND_TRUTH", labels),
            redirect_stdout(StringIO()) as output,
        ):
            experiment.main()
        self.assertIn("Top-3 labels: [0, 1, 1]", output.getvalue())
        self.assertIn("Total relevant chunks: 6", output.getvalue())
        self.assertIn("Precision@3: 0.67", output.getvalue())
        self.assertIn("Recall@3: 0.333", output.getvalue())

    def test_unknown_retrieved_id(self) -> None:
        chunks = [
            {"id": "unknown", "text": "test", "source": "sample.pdf", "page": 0, "distance": 1.0},
            {"id": self.ids[0], **REVIEWED_CHUNKS[self.ids[0]], "distance": 1.0},
            {"id": self.ids[1], **REVIEWED_CHUNKS[self.ids[1]], "distance": 1.0},
        ]
        with (
            patch.object(experiment, "open_invoice_collection", return_value=self.collection),
            patch.object(experiment, "search_invoice_chunks", return_value=chunks),
            redirect_stdout(StringIO()),
            self.assertRaisesRegex(ValueError, "has no reviewed label"),
        ):
            experiment.main()

    def test_changed_stored_text(self) -> None:
        self.records["documents"][0] = "Changed text"
        with (
            patch.object(experiment, "open_invoice_collection", return_value=self.collection),
            redirect_stdout(StringIO()),
            self.assertRaisesRegex(ValueError, "changed since manual review"),
        ):
            experiment.main()


if __name__ == "__main__":
    unittest.main()
