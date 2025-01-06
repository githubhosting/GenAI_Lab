# Question 1: Computing the TF-IDF Matrix using NumPy
# Task: Write a Python function to compute the TF-IDF matrix for the given set of documents using only NumPy.


import numpy as np


def compute_tf_idf(documents, vocabulary):
    N = len(documents)
    V = len(vocabulary)

    # Initialize TF matrix (N x V)
    tf = np.zeros((N, V))

    # Build term frequency matrix
    for i, doc in enumerate(documents):
        words = doc.lower().split()
        for word in words:
            if word in vocabulary:
                j = vocabulary.index(word)
                tf[i, j] += 1
        tf[i] = tf[i] / len(words)  # Normalize TF by document length

    # Compute Document Frequency (DF)
    df = np.zeros(V)
    for j, term in enumerate(vocabulary):
        df[j] = sum(1 for doc in documents if term in doc.lower().split())

    # Compute Inverse Document Frequency (IDF)
    idf = np.log(N / (df + 1))  # Add 1 to avoid division by zero

    # Compute TF-IDF matrix
    tf_idf = tf * idf  # Element-wise multiplication

    return tf_idf


# Example usage:
documents = [
    "cat sat on the mat",
    "dog sat on the log",
    "cat and dog played together"
]

vocabulary = list(set(" ".join(documents).lower().split()))
tf_idf_matrix = compute_tf_idf(documents, vocabulary)

print("Vocabulary:", vocabulary)
print("TF-IDF Matrix:\n", tf_idf_matrix)
