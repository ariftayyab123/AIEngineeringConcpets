import argparse
from functools import cache
from math import isfinite
from numbers import Real
from typing import TypedDict

from langchain_huggingface import HuggingFaceEmbeddings

from embeddings.embedding_experiment import MODEL_NAME, VECTOR_DIMENSIONS
from vectorstore.chroma_experiment import open_invoice_collection


DEFAULT_QUESTION = "How much was charged for the monthly membership?"


class InvoiceChunk(TypedDict):
    id: str
    text: str
    source: str
    page: int
    distance: float


@cache
def _embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cpu"})


def search_invoice_chunks(question: str, limit: int = 3) -> list[InvoiceChunk]:
    """Return ranked invoice chunks with their source metadata and distances."""
    question = question.strip()
    if not question:
        raise ValueError("Question must contain text.")
    if type(limit) is not int or limit < 1:
        raise ValueError("Limit must be a positive integer.")

    collection = open_invoice_collection()
    count = collection.count()
    if count == 0:
        return []

    query_vector = _embedding_model().embed_query(question)
    if len(query_vector) != VECTOR_DIMENSIONS or not all(map(isfinite, query_vector)):
        raise ValueError(f"Expected a query vector with {VECTOR_DIMENSIONS} finite values.")

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(limit, count),
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"]
    if not ids or not ids[0]:
        return []

    documents = results["documents"]
    metadata = results["metadatas"]
    distances = results["distances"]
    if not documents or not metadata or not distances:
        raise ValueError("Chroma did not return documents, metadata, and distances.")

    chunks: list[InvoiceChunk] = []
    for record_id, text, chunk_metadata, distance in zip(
        ids[0], documents[0], metadata[0], distances[0], strict=True
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record {record_id} is missing nonempty document text.")
        source = chunk_metadata.get("source") if chunk_metadata is not None else None
        page = chunk_metadata.get("page") if chunk_metadata is not None else None
        if not isinstance(source, str) or not source.strip() or type(page) is not int or page < 0:
            raise ValueError(f"Record {record_id} is missing valid source or page metadata.")
        if isinstance(distance, bool) or not isinstance(distance, Real) or not isfinite(distance):
            raise ValueError(f"Record {record_id} has a distance that is not a finite number.")
        chunks.append(
            {"id": record_id, "text": text, "source": source, "page": page, "distance": distance}
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the stored invoice chunks.")
    parser.add_argument("questions", nargs="*", default=[DEFAULT_QUESTION], metavar="QUESTION")
    questions = parser.parse_args().questions
    if any(not question.strip() for question in questions):
        parser.error("Each question must contain text.")

    collection = open_invoice_collection()
    count = collection.count()
    for question in questions:
        try:
            chunks = search_invoice_chunks(question)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"Question: {question}")
        print(f"Collection: {collection.name}")
        print(f"Stored records: {count}")
        if not chunks:
            print("No invoice chunks found.")
        for rank, chunk in enumerate(chunks, start=1):
            print(f"\nResult {rank}")
            print(f"ID: {chunk['id']}")
            print(f"Distance: {chunk['distance']:.6f}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Text: {chunk['text']}")
        print()


if __name__ == "__main__":
    main()
