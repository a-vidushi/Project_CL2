import random
from collections import Counter


def assign_document_folds(instances, number_of_folds, seed):
    """Map each document to a fold so that all its instances are tested together.

    Documents are sorted by size and dealt out in a snake order, so every fold
    gets a similar mix of large and small documents.
    """
    document_sizes = Counter(instance.document_id for instance in instances)
    documents = list(document_sizes)
    random.Random(seed).shuffle(documents)
    documents.sort(key=lambda document: document_sizes[document], reverse=True)

    fold_of_document = {}
    for position, document in enumerate(documents):
        round_number, offset = divmod(position, number_of_folds)
        fold_of_document[document] = offset if round_number % 2 == 0 else number_of_folds - 1 - offset
    return fold_of_document


def split_off_development_documents(documents, share=0.2, seed=0):
    """Pick a share of the distinct training documents as a development set for tuning."""
    unique_documents = sorted(set(documents))
    random.Random(seed).shuffle(unique_documents)
    development_count = max(1, round(len(unique_documents) * share))
    return set(unique_documents[:development_count])
