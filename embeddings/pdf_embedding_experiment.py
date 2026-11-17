from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.pdf_loader import DEFAULT_PDF, load_pdf_pages
from ingestion.text_splitter import split_pdf_pages

from .embedding_experiment import MODEL_NAME, VECTOR_DIMENSIONS


EXPECTED_CHUNKS = 8


def main() -> None:
    pages = load_pdf_pages(DEFAULT_PDF)
    chunks = split_pdf_pages(pages)
    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(f"Expected {EXPECTED_CHUNKS} invoice chunks, found {len(chunks)}.")

    model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cpu"})
    vectors = model.embed_documents([chunk.page_content for chunk in chunks])

    if len(vectors) != len(chunks):
        raise ValueError(f"Expected {len(chunks)} vectors, received {len(vectors)}.")
    for index, vector in enumerate(vectors, start=1):
        if len(vector) != VECTOR_DIMENSIONS:
            raise ValueError(f"Vector {index} has {len(vector)} dimensions. Expected {VECTOR_DIMENSIONS}.")

    embedded_chunks = list(zip(chunks, vectors, strict=True))
    print(f"Original documents: {len(pages)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Embedding shape: {len(embedded_chunks)} x {len(vectors[0])}")
    for index, (chunk, vector) in enumerate(embedded_chunks, start=1):
        metadata = chunk.metadata
        print(
            f"Vector {index}: {len(vector)} dimensions | "
            f"page: {metadata['page']} | start_index: {metadata['start_index']} | "
            f"source: {metadata['source']}"
        )


if __name__ == "__main__":
    main()
