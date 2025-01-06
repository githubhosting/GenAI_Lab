# Question 5: Creating a Word Embedding Matrix with Pre-trained Embeddings
# Task:
# 1. Implement the function create_embedding_matrix_with_pretrained(corpus, pretrained_embeddings, embedding_dim).
# 2. Test the function with the given corpus and pre-trained embeddings.

import numpy as np


def create_embedding_matrix_with_pretrained(corpus, pretrained_embeddings, embedding_dim):
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
    # Initialize embedding matrix
    E = np.zeros((V, embedding_dim))

    # Assign embeddings
    for word, idx in vocabulary.items():
        if word in pretrained_embeddings:
            E[idx] = np.array(pretrained_embeddings[word])
        else:
            E[idx] = np.random.rand(embedding_dim)  # Random initialization

    # Define get_word_vector function
    def get_word_vector(word):
        word = word.lower()
        if word in vocabulary:
            idx = vocabulary[word]
            return E[idx]
        else:
            return np.zeros(embedding_dim)

    return E, vocabulary, get_word_vector


# Example usage:
corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]

pretrained_embeddings = {
    "machine": [0.1, 0.2, 0.3],
    "learning": [0.2, 0.3, 0.4],
    "amazing": [0.3, 0.4, 0.5],
    "love": [0.4, 0.5, 0.6]
}

embedding_dim = 3

E, vocabulary, get_word_vector = create_embedding_matrix_with_pretrained(
    corpus, pretrained_embeddings, embedding_dim)

print("Vocabulary:", vocabulary)
print("Embedding Matrix E:\n", E)

# Test get_word_vector
word = "machine"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)

word = "i"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)  # Randomly initialized

word = "unknown"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)  # Returns zeros
