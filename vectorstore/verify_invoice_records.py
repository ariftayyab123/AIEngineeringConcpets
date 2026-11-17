from .chroma_experiment import open_invoice_collection


def main() -> None:
    collection = open_invoice_collection()
    count = collection.count()
    if count != 8:
        raise SystemExit(f"Expected 8 stored records, found {count}.")

    record_id = "sample-p0-c0"
    result = collection.get(ids=[record_id], include=["documents", "metadatas", "embeddings"])
    if result["ids"] != [record_id]:
        raise SystemExit(f"Expected exactly one matching record for {record_id}.")

    documents = result["documents"]
    metadata = result["metadatas"]
    embeddings = result["embeddings"]
    if not documents or not documents[0]:
        raise SystemExit(f"Missing text for {record_id}.")
    if not metadata or metadata[0] != {"source": "sample.pdf", "page": 0, "start_index": 0}:
        raise SystemExit(f"Unexpected metadata for {record_id}: {metadata}")
    if embeddings is None or len(embeddings) != 1 or len(embeddings[0]) != 384:
        raise SystemExit(f"Expected one 384-dimensional embedding for {record_id}.")

    print(f"Stored records after restart: {count}")
    print(f"Record ID: {result['ids'][0]}")
    print(f"Text preview: {' '.join(documents[0].split())[:160]}")
    print(f"Metadata: {metadata[0]}")
    print(f"Embedding dimensions: {len(embeddings[0])}")


if __name__ == "__main__":
    main()
