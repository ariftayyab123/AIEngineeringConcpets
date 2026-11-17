import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf.errors import PdfReadError


DEFAULT_PDF = Path(__file__).resolve().parents[1] / "data" / "sample.pdf"


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    loader = PyPDFLoader(file_path=str(pdf_path), mode="page")
    return loader.load()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and inspect PDF pages.")
    parser.add_argument("pdf_path", nargs="?", type=Path, default=DEFAULT_PDF)
    pdf_path = parser.parse_args().pdf_path

    try:
        if not pdf_path.is_file():
            print(f"Error: No PDF file found at '{pdf_path}'.", file=sys.stderr)
            return 1

        if pdf_path.suffix.lower() != ".pdf":
            print(f"Error: Expected a .pdf file: '{pdf_path}'.", file=sys.stderr)
            return 1

        documents = load_pdf_pages(pdf_path)
    except (OSError, ValueError, PdfReadError) as error:
        print(f"Error: Could not read PDF '{pdf_path}': {error}", file=sys.stderr)
        return 1

    print(f"Total documents: {len(documents)}")
    if documents:
        first_document = documents[0]
        print("\nDocument 1")
        print("Content: |")
        for line in first_document.page_content.splitlines():
            print(f"  {line}")
        print("Metadata:")
        for key, value in first_document.metadata.items():
            print(f"  {key}: {value}")

    for page_number, document in enumerate(documents, start=1):
        if not document.page_content.strip():
            print(f"Warning: No text extracted from page {page_number}.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
