import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import PdfReadError

if __package__:
    from .pdf_loader import DEFAULT_PDF, load_pdf_pages
else:
    from pdf_loader import DEFAULT_PDF, load_pdf_pages


CHUNK_SIZE = 500


def split_pdf_pages(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def main() -> int:
    try:
        documents = load_pdf_pages(DEFAULT_PDF)
    except (OSError, ValueError, PdfReadError) as error:
        print(f"Error: Could not read PDF '{DEFAULT_PDF}': {error}", file=sys.stderr)
        return 1

    chunks = split_pdf_pages(documents)
    print(f"Original documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    pages_by_source = {
        (document.metadata["source"], document.metadata["page"]): document
        for document in documents
    }
    for chunk_number, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        parent = pages_by_source.get((metadata.get("source"), metadata.get("page")))
        if parent is None:
            print(f"Error: Chunk {chunk_number} has invalid source or page metadata.", file=sys.stderr)
            return 1

        start_index = metadata["start_index"]
        character_count = len(chunk.page_content)
        if character_count > CHUNK_SIZE:
            print(f"Error: Chunk {chunk_number} exceeds {CHUNK_SIZE} characters.", file=sys.stderr)
            return 1

        if start_index >= 0:
            original_text = parent.page_content[start_index : start_index + character_count]
            content_matches = original_text == chunk.page_content
        else:
            content_matches = chunk.page_content in parent.page_content
        if not content_matches:
            print(f"Error: Chunk {chunk_number} does not match its source page.", file=sys.stderr)
            return 1

        print(f"\nChunk {chunk_number}")
        print(f"Length: {character_count}")
        print(f"Page index: {metadata['page']}")
        print(f"Start index: {start_index}")
        print(f"Source: {Path(metadata['source']).name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
