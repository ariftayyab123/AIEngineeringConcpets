import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSIONS = 384


def main() -> None:
    texts = [
        "The customer paid 37.50 dollars.",
        "The invoice payment amount was $37.50.",
        "The employee works remotely.",
    ]
    query = "How much did the customer pay?"

    model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cpu"})
    document_vectors = np.asarray(model.embed_documents(texts), dtype=np.float64)
    query_vector = np.asarray(model.embed_query(query), dtype=np.float64)

    if document_vectors.shape != (len(texts), VECTOR_DIMENSIONS):
        raise ValueError(f"Unexpected document vector shape: {document_vectors.shape}")
    if query_vector.shape != (VECTOR_DIMENSIONS,):
        raise ValueError(f"Unexpected query vector shape: {query_vector.shape}")

    document_norms = np.linalg.norm(document_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    if np.any(document_norms == 0) or query_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    similarities = (document_vectors @ query_vector) / (document_norms * query_norm)
    ranking = np.argsort(-similarities, kind="stable")

    print(f"Document vectors: {document_vectors.shape[0]}")
    print(f"Vector dimensions: {document_vectors.shape[1]}")
    print(f"Query dimensions: {query_vector.size}")
    print("\nSimilarity results:")
    for index in ranking:
        print(f"Document {index + 1}: {similarities[index]:.6f} | {texts[index]}")

    best_index = ranking[0]
    print(f"\nMost similar document: Document {best_index + 1} | {texts[best_index]}")


if __name__ == "__main__":
    main()
