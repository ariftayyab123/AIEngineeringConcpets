from langchain_community.document_loaders import PyPDFLoader

from pdf_loader import DEFAULT_PDF


def main() -> None:
    loader = PyPDFLoader(file_path=str(DEFAULT_PDF), mode="page")
    total_pages = 0

    for document in loader.lazy_load():
        page_index = document.metadata["page"]
        character_count = len(document.page_content)
        print(f"Page index: {page_index} | Characters: {character_count}")
        total_pages += 1

    print(f"\nTotal processed pages: {total_pages}")


if __name__ == "__main__":
    main()
