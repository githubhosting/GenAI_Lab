# Question 4: Creating a Word Embedding Matrix
# Task:
# 1. Implement the function create_embedding_matrix(corpus, embedding_dim).
# 2. Test the function and get_word_vector with the given corpus and embedding_dim=3.


import numpy as np


def create_embedding_matrix(corpus, embedding_dim):
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
    # Initialize embedding matrix with random values between 0 and 1
    E = np.random.rand(V, embedding_dim)

    # Create word to index mapping (already done in vocabulary)
    word_to_index = vocabulary

    # Define get_word_vector function
    def get_word_vector(word):
        word = word.lower()
        if word in word_to_index:
            idx = word_to_index[word]
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
embedding_dim = 3

E, vocabulary, get_word_vector = create_embedding_matrix(corpus, embedding_dim)

print("Vocabulary:", vocabulary)
print("Embedding Matrix E:\n", E)

# Test get_word_vector
word = "learning"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)

# Test with a word not in the vocabulary
word = "unknown"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)
