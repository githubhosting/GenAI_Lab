# Question 6: Generating One-Hot Encodings
# Task:
# 1. Implement the function create_one_hot_encodings(corpus).
# 2. Test the function with the given corpus.


import numpy as np


def create_one_hot_encodings(corpus):
    # Preprocessing
    vocabulary = {}
    index = 0
    for sentence in corpus:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1

    V = len(vocabulary)
    # Initialize one-hot encoding matrix
    one_hot_encodings = {}

    for word, idx in vocabulary.items():
        one_hot_vector = np.zeros(V)
        one_hot_vector[idx] = 1
        one_hot_encodings[word] = one_hot_vector

    return vocabulary, one_hot_encodings


# Example usage:
corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]

vocabulary, one_hot_encodings = create_one_hot_encodings(corpus)

print("Vocabulary:", vocabulary)
print("\nOne-Hot Encodings:")
for word, one_hot_vector in one_hot_encodings.items():
    print(f"Word: '{word}' - One-Hot Vector: {one_hot_vector}")
