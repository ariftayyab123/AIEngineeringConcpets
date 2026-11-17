from langchain.tools import tool

from retrieval.semantic_search import search_invoice_chunks


@tool
def search_invoices(query: str) -> str:
    """Search indexed invoices and return source evidence for invoice questions."""
    query = query.strip()
    if not query:
        raise ValueError("Query must contain text.")

    chunks = search_invoice_chunks(query)
    if not chunks:
        return "No invoice evidence found."

    results = []
    for rank, chunk in enumerate(chunks, start=1):
        results.append(
            f"Result {rank}\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page'] + 1}\n"
            f"Chunk ID: {chunk['id']}\n"
            f"Distance: {chunk['distance']:.6f}\n\n"
            f"Evidence:\n{chunk['text']}"
        )
    return "\n\n".join(results)


def main() -> None:
    print(search_invoices.invoke({
        "query": "How much was charged for the monthly membership?",
    }))


if __name__ == "__main__":
    main()
