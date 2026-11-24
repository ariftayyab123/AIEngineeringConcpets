"""Compare retrieval distances with manual relevance labels for two questions."""

from hashlib import sha256

from retrieval.semantic_search import search_invoice_chunks
from vectorstore.chroma_experiment import open_invoice_collection

from .retrieval_metrics import precision_at_k, recall_at_k

# This fingerprint identifies the full text inspected for the manual labels below.
# A changed or unseen chunk stays unreviewed, even if its ID is reused.
REVIEWED_MEMBERSHIP_TEXT = "4eb1ab41f891b7813d17b2745630e76845c3c7a0b4b4d4565bf5842ecd57095c"
REVIEWED_INVOICE_TEXT = "157b88aa8956439d49ac655e7c901f753cca635bc202519e7c3bac12926030e4"
# All eight records were individually inspected. Duplicate text still counts
# separately because recall is measured over stored chunk IDs.
REFERENCE_CHUNKS = {
    "sample-p0-c0": REVIEWED_INVOICE_TEXT,
    "sample-p0-c1": REVIEWED_MEMBERSHIP_TEXT,
    "sample-p1-c0": REVIEWED_INVOICE_TEXT,
    "sample-p1-c1": REVIEWED_MEMBERSHIP_TEXT,
    "sample-p2-c0": REVIEWED_INVOICE_TEXT,
    "sample-p2-c1": REVIEWED_MEMBERSHIP_TEXT,
    "sample-p3-c0": REVIEWED_INVOICE_TEXT,
    "sample-p3-c1": REVIEWED_MEMBERSHIP_TEXT,
}
CASES = [
    {
        "id": "T07-membership",
        "question": "How much was charged for the monthly membership?",
        "labels": {
            REVIEWED_INVOICE_TEXT: (
                True, "Contains the monthly membership description, discounted total, and USD $37.50.",
            ),
            REVIEWED_MEMBERSHIP_TEXT: (
                True, "Contains Premium Membership - Monthly and USD $37.50.",
            ),
        },
    },
    {
        "id": "T07-weather",
        "question": "What is the weather in London?",
        "labels": {
            REVIEWED_INVOICE_TEXT: (
                False, "Invoice amounts and identifiers provide no London weather evidence.",
            ),
            REVIEWED_MEMBERSHIP_TEXT: (
                False, "Invoice membership details provide no London weather evidence.",
            ),
        },
    },
]


def main() -> None:
    print("T07: Retrieval observation with manual relevance labels")
    print("Page values below are zero-based metadata indexes.")
    print("Labels apply only to the exact reviewed text for each question.")
    records = open_invoice_collection().get(include=["documents"])
    documents = records["documents"]
    if documents is None:
        raise ValueError("Collection did not return document text for reference validation.")
    current_chunks = {
        record_id: sha256(text.encode("utf-8")).hexdigest()
        for record_id, text in zip(records["ids"], documents, strict=True)
    }
    if current_chunks != REFERENCE_CHUNKS:
        raise ValueError("Collection differs from the eight reviewed records. Review the reference labels again.")

    summaries: list[str] = []
    for case in CASES:
        chunks = search_invoice_chunks(case["question"])
        print(f"\nTest: {case['id']}")
        print(f"Question: {case['question']}")
        reference_labels = {
            record_id: int(case["labels"][fingerprint][0])
            for record_id, fingerprint in REFERENCE_CHUNKS.items()
        }
        total_relevant = sum(reference_labels.values())
        print(f"Manual collection labels by chunk ID: {reference_labels}")
        print(f"Total relevant in reviewed collection: {total_relevant}")
        supported = 0
        reviewed = 0
        relevance_labels: list[int] = []
        for rank, chunk in enumerate(chunks, start=1):
            fingerprint = sha256(chunk["text"].encode("utf-8")).hexdigest()
            label = case["labels"].get(fingerprint)
            print(f"\nResult {rank}")
            print(f"Chunk ID: {chunk['id']}")
            print(f"Page: {chunk['page']}")
            print(f"Distance: {chunk['distance']:.6f}")
            print(f"Text preview: {' '.join(chunk['text'].split())[:160]}")
            if label is None:
                print("Supports question: UNREVIEWED (requires manual inspection)")
            else:
                supports_question, reason = label
                reviewed += 1
                supported += supports_question
                relevance_labels.append(int(supports_question))
                print(f"Supports question: {'YES' if supports_question else 'NO'} (manual)")
                print(f"Reason: {reason}")

        if reviewed == len(chunks) and len(chunks) >= 3:
            print(f"Manual relevance labels (retrieval order): {relevance_labels}")
            precision_report = f"Precision@3: {precision_at_k(relevance_labels, 3):.2f}"
            if total_relevant == 0:
                recall_report = "Recall@3: undefined (no relevant chunks in the collection)"
            else:
                recall_report = f"Recall@3: {recall_at_k(relevance_labels, 3, total_relevant):.3f}"
        else:
            precision_report = "Precision@3: unavailable (requires three results with manual labels)"
            recall_report = "Recall@3: unavailable (requires three results with manual labels)"
        print(precision_report)
        print(recall_report)

        if chunks:
            distances = [chunk["distance"] for chunk in chunks]
            distance_range = f"{min(distances):.6f} to {max(distances):.6f}"
        else:
            distance_range = "no results"
        summaries.append(
            f"{case['id']}: distances {distance_range}; "
            f"{supported}/{reviewed} reviewed chunks support the question; "
            f"{len(chunks) - reviewed} unreviewed; {precision_report}; {recall_report}."
        )

    print("\nDistance and manual relevance comparison:")
    for summary in summaries:
        print(summary)
    print("\nRecorded observations for the reviewed sample:")
    print("The same repeated membership text was returned for both questions.")
    print("It supports the membership charge and provides no weather evidence.")
    print("Compare raw distances with these manual labels. Two questions do not establish a relevance threshold.")


if __name__ == "__main__":
    main()
