# Question 2: Generating n-grams for a Sentence
# Task: Write a Python function to generate n-grams for a given sentence.


def generate_ngrams(sentence, n):
    words = sentence.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngrams.append(ngram)
    return ngrams


# Example usage:
sentence = "The quick brown fox jumps over the lazy dog."
n = 3
ngrams = generate_ngrams(sentence, n)
print(f"{n}-grams:")
for gram in ngrams:
    print(gram)

# Output:
# 3-grams:
# ('the', 'quick', 'brown')
# ('quick', 'brown', 'fox')
# ('brown', 'fox', 'jumps')
# ('fox', 'jumps', 'over')
# ('jumps', 'over', 'the')
# ('over', 'the', 'lazy')
# ('the', 'lazy', 'dog.')