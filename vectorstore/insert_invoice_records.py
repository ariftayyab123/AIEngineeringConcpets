from embeddings.embed_pdf_chunks import generate_invoice_records

from .chroma_experiment import open_invoice_collection


EXPECTED_RECORD_COUNT = 8


def main() -> None:
    collection = open_invoice_collection()
    existing_count = collection.count()
    if existing_count != 0:
        raise SystemExit(
            f"Collection '{collection.name}' already contains {existing_count} records. "
            "Initial insertion requires an empty collection."
        )

    _, records = generate_invoice_records()
    ids = [record["chunk_id"] for record in records]
    texts = [record["text"] for record in records]
    vectors = [record["embedding"] for record in records]
    metadata = [dict(record["metadata"]) for record in records]

    counts = (len(ids), len(texts), len(vectors), len(metadata))
    if counts != (EXPECTED_RECORD_COUNT,) * 4:
        raise SystemExit(f"Expected eight IDs, texts, vectors, and metadata entries. Got {counts}.")
    if len(set(ids)) != len(ids):
        raise SystemExit("Pre-insertion validation failed: chunk IDs must be unique.")

    print(f"Collection: {collection.name}")
    print(f"Prepared records: {len(ids)}")
    print(f"Embedding dimensions: {len(vectors[0])}")
    print("\nPre-insertion validation: PASSED")
    collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadata)
    stored_count = collection.count()
    if stored_count != EXPECTED_RECORD_COUNT:
        raise SystemExit(f"Expected {EXPECTED_RECORD_COUNT} stored records, found {stored_count}.")

    print(f"Stored records: {stored_count}")


if __name__ == "__main__":
    main()
