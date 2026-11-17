import sys
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf.errors import PdfReadError

from ingestion.pdf_loader import DEFAULT_PDF, load_pdf_pages
from ingestion.text_splitter import split_pdf_pages

from .embedding_experiment import MODEL_NAME, VECTOR_DIMENSIONS


class ChunkMetadata(TypedDict):
    source: str
    page: int
    start_index: int


class ChunkRecord(TypedDict):
    chunk_id: str
    text: str
    embedding: list[float]
    metadata: ChunkMetadata


def build_records(chunks: list[Document], vectors: list[list[float]]) -> list[ChunkRecord]:
    if len(vectors) != len(chunks):
        raise ValueError(f"Expected {len(chunks)} vectors, received {len(vectors)}.")

    records: list[ChunkRecord] = []
    chunks_per_page: Counter[tuple[str, int]] = Counter()
    for number, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
        if len(vector) != VECTOR_DIMENSIONS:
            raise ValueError(f"Vector {number} has {len(vector)} dimensions. Expected {VECTOR_DIMENSIONS}.")
        if not all(type(value) in (int, float) and isfinite(value) for value in vector):
            raise ValueError(f"Vector {number} contains a value that is not a finite number.")

        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        start_index = chunk.metadata.get("start_index")
        if not isinstance(source, str) or not source or type(page) is not int or page < 0:
            raise ValueError(f"Chunk {number} is missing valid source or page metadata.")
        if type(start_index) is not int:
            raise ValueError(f"Chunk {number} is missing an integer start_index.")

        source_path = Path(source)
        page_key = (source, page)
        chunk_index = chunks_per_page[page_key]
        chunks_per_page[page_key] += 1
        records.append({
            "chunk_id": f"{source_path.stem}-p{page}-c{chunk_index}",
            "text": chunk.page_content,
            "embedding": vector,
            "metadata": {"source": source_path.name, "page": page, "start_index": start_index},
        })

    return records


def generate_invoice_records() -> tuple[int, list[ChunkRecord]]:
    pages = load_pdf_pages(DEFAULT_PDF)
    chunks = split_pdf_pages(pages)
    if not chunks:
        raise ValueError("The PDF produced no text chunks.")
    texts = [chunk.page_content for chunk in chunks]
    model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cpu"})
    vectors = model.embed_documents(texts)
    return len(pages), build_records(chunks, vectors)


def main() -> int:
    try:
        page_count, records = generate_invoice_records()
    except (OSError, ValueError, PdfReadError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1

    print(f"Original pages: {page_count}")
    print(f"Generated chunks: {len(records)}")
    print(f"Generated embeddings: {len(records)}")
    print(f"Embedding dimensions: {len(records[0]['embedding'])}")

    for number, record in enumerate(records, start=1):
        metadata = record["metadata"]
        print(f"\nRecord {number}")
        print(f"Chunk ID: {record['chunk_id']}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Start index: {metadata['start_index']}")
        print(f"Text length: {len(record['text'])}")
        print(f"Vector dimensions: {len(record['embedding'])}")

    print("\nValidation: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
