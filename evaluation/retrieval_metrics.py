def _validate_labels_and_k(relevance_labels: list[int], k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > len(relevance_labels):
        raise ValueError("k must not exceed the number of labels.")
    if any(type(label) is not int or label not in (0, 1) for label in relevance_labels):
        raise ValueError("Each relevance label must be an integer equal to 0 or 1.")


def precision_at_k(relevance_labels: list[int], k: int) -> float:
    """Return the fraction of relevant results in the first k positions."""
    _validate_labels_and_k(relevance_labels, k)
    first_k_labels = relevance_labels[:k]
    relevant_count = sum(first_k_labels)
    return relevant_count / k


def recall_at_k(relevance_labels: list[int], k: int, total_relevant: int) -> float:
    """Return the fraction of all relevant chunks found in the first k results."""
    _validate_labels_and_k(relevance_labels, k)
    if type(total_relevant) is not int or total_relevant < 0:
        raise ValueError("total_relevant must be a nonnegative integer.")
    if total_relevant == 0:
        raise ValueError("Recall is undefined when total_relevant is zero.")
    if sum(relevance_labels) > total_relevant:
        raise ValueError("Relevant labels across all supplied results exceed total_relevant.")
    return sum(relevance_labels[:k]) / total_relevant
