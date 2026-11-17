import argparse
import re
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from pypdf.errors import PdfReadError


EXPECTED_INVOICE = "13594027"
EXPECTED_AMOUNT = "37.50"
EXPECTED_STATUSES = ("PAID", "UNPAID", "REFUNDED", "CANCELLED")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate extraction from a four-page invoice PDF.")
    parser.add_argument("pdf_path", type=Path)
    pdf_path = parser.parse_args().pdf_path

    try:
        if not pdf_path.is_file():
            print(f"Error: No PDF file found at '{pdf_path}'.", file=sys.stderr)
            return 1
        if pdf_path.suffix.lower() != ".pdf":
            print(f"Error: Expected a .pdf file: '{pdf_path}'.", file=sys.stderr)
            return 1

        loader = PyPDFLoader(file_path=str(pdf_path), mode="page")
        documents = loader.load()
    except (OSError, ValueError, PdfReadError) as error:
        print(f"Error: Could not read PDF '{pdf_path}': {error}", file=sys.stderr)
        return 1

    if len(documents) != len(EXPECTED_STATUSES):
        print(
            f"Error: Expected {len(EXPECTED_STATUSES)} pages, found {len(documents)}.",
            file=sys.stderr,
        )
        return 1

    first_page_text = documents[0].page_content
    for document in documents:
        text = document.page_content
        page_index = document.metadata["page"]
        expected_status = EXPECTED_STATUSES[page_index]
        # Whole-word matching keeps PAID from matching UNPAID.
        status_found = re.search(rf"\b{expected_status}\b", text.upper()) is not None

        print(f"Page: {page_index + 1}")
        print(f"Characters: {len(text)}")
        print(f"Invoice number found: {EXPECTED_INVOICE in text}")
        print(f"Amount found: {EXPECTED_AMOUNT in text}")
        print(f"Expected status: {expected_status}")
        print(f"Expected status found: {status_found}")
        print(f"Text identical to page 1: {text == first_page_text}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
