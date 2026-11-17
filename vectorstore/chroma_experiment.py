from pathlib import Path

import chromadb


CHROMA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
COLLECTION_NAME = "invoice_chunks"


def open_invoice_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIRECTORY)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=None)


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIRECTORY)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )
    print(f"Storage: {CHROMA_DIRECTORY}")
    print(f"Collection: {collection.name}")
    print(f"Stored records: {collection.count()}")


if __name__ == "__main__":
    main()
