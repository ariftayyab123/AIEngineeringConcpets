from .verify_chroma_collection import read_collection_status


def main() -> None:
    name, count = read_collection_status()
    print("Persistence verification: PASSED")
    print(f"Collection: {name}")
    print(f"Stored records: {count}")


if __name__ == "__main__":
    main()
