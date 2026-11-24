"""Review the full collection and measure membership retrieval against chunk IDs."""

from retrieval.semantic_search import search_invoice_chunks
from vectorstore.chroma_experiment import open_invoice_collection

from .ground_truth import GROUND_TRUTH, QUESTION, RELEVANCE_RULE, REVIEWED_CHUNKS
from .retrieval_metrics import precision_at_k, recall_at_k


def main() -> None:
    collection = open_invoice_collection()
    records = collection.get(include=["documents", "metadatas"])
    ids = records["ids"]
    documents = records["documents"]
    metadata = records["metadatas"]
    if documents is None or metadata is None:
        raise ValueError("Evaluation error: stored text and metadata are required.")
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("Evaluation error: expected exactly eight unique stored chunk IDs.")
    if set(ids) != set(GROUND_TRUTH) or set(ids) != set(REVIEWED_CHUNKS):
        raise ValueError("Evaluation error: unexpected or missing ground-truth IDs.")
    if any(type(label) is not int or label not in (0, 1) for label in GROUND_TRUTH.values()):
        raise ValueError("Evaluation error: ground-truth labels must be integer 0 or 1.")

    print(f"Question: {QUESTION}")
    print(f"Relevance rule: {RELEVANCE_RULE}")
    print("Review unit: stored chunk ID, including repeated text.")
    for record_id, text, chunk_metadata in zip(ids, documents, metadata, strict=True):
        reviewed = REVIEWED_CHUNKS[record_id]
        if (
            text != reviewed["text"] or chunk_metadata is None
            or chunk_metadata.get("source") != reviewed["source"]
            or chunk_metadata.get("page") != reviewed["page"]
        ):
            raise ValueError(f"Evaluation error: {record_id} changed since manual review.")
        print(f"\nChunk ID: {record_id}")
        print(f"Source: {reviewed['source']}")
        print(f"Page index: {reviewed['page']}")
        print(f"Complete text:\n{text}")
        print(f"Manual relevance label: {GROUND_TRUTH[record_id]}")

    chunks = search_invoice_chunks(QUESTION, limit=3)
    if len(chunks) != 3 or len({chunk["id"] for chunk in chunks}) != 3:
        raise ValueError("Evaluation error: expected three distinct retrieved chunk IDs.")
    ranked_labels: list[int] = []
    print("\nRanked retrieval:")
    for rank, chunk in enumerate(chunks, start=1):
        record_id = chunk["id"]
        if record_id not in GROUND_TRUTH:
            raise ValueError(f"Evaluation error: retrieved ID {record_id} has no reviewed label.")
        reviewed = REVIEWED_CHUNKS[record_id]
        if (
            chunk["text"] != reviewed["text"] or chunk["source"] != reviewed["source"]
            or chunk["page"] != reviewed["page"]
        ):
            raise ValueError(f"Evaluation error: retrieved {record_id} differs from the reviewed record.")
        label = GROUND_TRUTH[record_id]
        ranked_labels.append(label)
        print(f"{rank}. ID: {record_id} | Label: {label} | Distance: {chunk['distance']:.6f}")

    total_relevant = sum(GROUND_TRUTH.values())
    print("\nEvaluation report:")
    print(f"Total stored chunks: {len(ids)}")
    print(f"Total relevant chunks: {total_relevant}")
    print(f"Top-3 labels: {ranked_labels}")
    print(f"Precision@3: {precision_at_k(ranked_labels, 3):.2f}")
    if total_relevant == 0:
        print("Recall@3: undefined (no relevant chunks)")
    else:
        print(f"Recall@3: {recall_at_k(ranked_labels, 3, total_relevant):.3f}")


if __name__ == "__main__":
    main()
