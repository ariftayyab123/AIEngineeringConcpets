from .chroma_experiment import open_invoice_collection


def read_collection_status() -> tuple[str, int]:
    collection = open_invoice_collection()
    return collection.name, collection.count()


def main() -> None:
    name, count = read_collection_status()
    print(f"Collection: {name}")
    print(f"Stored records: {count}")


if __name__ == "__main__":
    main()
