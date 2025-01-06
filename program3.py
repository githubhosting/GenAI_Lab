# Question 3: Computing a 3-gram Language Model
# Task: Write a Python function to compute a 3-gram language model.


def compute_trigram_language_model(documents):
    from collections import defaultdict

    trigram_counts = defaultdict(int)
    total_trigrams = 0

    for doc in documents:
        words = doc.lower().split()
        for i in range(len(words) - 2):
            trigram = tuple(words[i:i + 3])
            trigram_counts[trigram] += 1
            total_trigrams += 1

    # Compute probabilities
    trigram_probabilities = {}
    for trigram, count in trigram_counts.items():
        trigram_probabilities[trigram] = count / total_trigrams

    return trigram_probabilities


# Example usage:
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The quick blue fox jumps over the lazy cat",
    "The lazy dog sleeps under the blue sky"
]

trigram_model = compute_trigram_language_model(documents)

print("Trigram Probabilities:")
for trigram, prob in trigram_model.items():
    print(f"{trigram}: {prob}")
