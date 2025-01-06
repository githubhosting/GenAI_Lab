## Question 1: Computing the TF-IDF Matrix using NumPy

## Task: Write a Python function to compute the TF-IDF matrix for the given set of documents using only NumPy.

```py
import numpy as np

def compute_tf_idf(documents, vocabulary):
    N = len(documents)
    V = len(vocabulary)
    tf = np.zeros((N, V))

    for i, doc in enumerate(documents):
        words = doc.lower().split()
        for word in words:
            if word in vocabulary:
                j = vocabulary.index(word)
                tf[i, j] += 1
        tf[i] = tf[i] / len(words)

    df = np.zeros(V)
    for j, term in enumerate(vocabulary):
        df[j] = sum(1 for doc in documents if term in doc.lower().split())

    idf = np.log(N / (df + 1))

    tf_idf = tf * idf

    return tf_idf

documents = [
    "cat sat on the mat",
    "dog sat on the log",
    "cat and dog played together"
]

vocabulary = list(set(" ".join(documents).lower().split()))
tf_idf_matrix = compute_tf_idf(documents, vocabulary)

print("Vocabulary:", vocabulary)
print("TF-IDF Matrix:\n", tf_idf_matrix)
```

## Question 2: Generating n-grams for a Sentence

## Task: Write a Python function to generate n-grams for a given sentence.

```py
def generate_ngrams(sentence, n):
    words = sentence.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i + n])
        ngrams.append(ngram)
    return ngrams

sentence = "The quick brown fox jumps over the lazy dog."
n = 3
ngrams = generate_ngrams(sentence, n)
print(f"{n}-grams:")
for gram in ngrams:
    print(gram)
```

## Question 3: Computing a 3-gram Language Model

## Task: Write a Python function to compute a 3-gram language model.

```py
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

    trigram_probabilities = {}
    for trigram, count in trigram_counts.items():
        trigram_probabilities[trigram] = count / total_trigrams

    return trigram_probabilities


documents = [
    "The quick brown fox jumps over the lazy dog",
    "The quick blue fox jumps over the lazy cat",
    "The lazy dog sleeps under the blue sky"
]

trigram_model = compute_trigram_language_model(documents)

print("Trigram Probabilities:")
for trigram, prob in trigram_model.items():
    print(f"{trigram}: {prob}")
```

## Question 4: Creating a Word Embedding Matrix

### 1. Implement the function create_embedding_matrix(corpus, embedding_dim).

### 2. Test the function and get_word_vector with the given corpus and embedding_dim=3.

```py
import numpy as np

def create_embedding_matrix(corpus, embedding_dim):
    vocabulary = {}
    index = 0
    for sentence in corpus:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1

    V = len(vocabulary)
    E = np.random.rand(V, embedding_dim)

    word_to_index = vocabulary

    def get_word_vector(word):
        word = word.lower()
        if word in word_to_index:
            idx = word_to_index[word]
            return E[idx]
        else:
            return np.zeros(embedding_dim)

    return E, vocabulary, get_word_vector


corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]
embedding_dim = 3

E, vocabulary, get_word_vector = create_embedding_matrix(corpus, embedding_dim)

print("Vocabulary:", vocabulary)
print("Embedding Matrix E:\n", E)

word = "learning"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)

word = "unknown"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)
```

## Question 5: Creating a Word Embedding Matrix with Pre-trained Embeddings

### 1. Implement the function create_embedding_matrix_with_pretrained(corpus, pretrained_embeddings, embedding_dim).

### 2. Test the function with the given corpus and pre-trained embeddings.

```py
import numpy as np

def create_embedding_matrix_with_pretrained(corpus, pretrained_embeddings, embedding_dim):
    vocabulary = {}
    index = 0
    for sentence in corpus:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1

    V = len(vocabulary)
    E = np.zeros((V, embedding_dim))

    for word, idx in vocabulary.items():
        if word in pretrained_embeddings:
            E[idx] = np.array(pretrained_embeddings[word])
        else:
            E[idx] = np.random.rand(embedding_dim)

    def get_word_vector(word):
        word = word.lower()
        if word in vocabulary:
            idx = vocabulary[word]
            return E[idx]
        else:
            return np.zeros(embedding_dim)

    return E, vocabulary, get_word_vector


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

word = "machine"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)

word = "i"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)  # Randomly initialized

word = "unknown"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)  # Returns zeros
```

## Question 6: Generating One-Hot Encodings

### 1. Implement the function create_one_hot_encodings(corpus).

### 2. Test the function with the given corpus.

```py
import numpy as np

def create_one_hot_encodings(corpus):
    vocabulary = {}
    index = 0
    for sentence in corpus:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1

    V = len(vocabulary)
    one_hot_encodings = {}

    for word, idx in vocabulary.items():
        one_hot_vector = np.zeros(V)
        one_hot_vector[idx] = 1
        one_hot_encodings[word] = one_hot_vector

    return vocabulary, one_hot_encodings

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
```

## Question 7: Implementing the Skip-Gram Model

### 1. Implement the function generate_skip_gram_pairs(sentences, window_size).

### 2. Test it with the given sentences and window_size = 2.

```py
def generate_skip_gram_pairs(sentences, window_size):
    vocabulary = {}
    index = 0
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1

    training_pairs = []
    for sentence in sentences:
        words = sentence.lower().split()
        for i, target_word in enumerate(words):
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_word = words[j]
                    training_pairs.append((target_word, context_word))

    return vocabulary, training_pairs


sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]

window_size = 2

vocabulary, training_pairs = generate_skip_gram_pairs(sentences, window_size)

print("Vocabulary:", vocabulary)
print("\nSkip-Gram Training Pairs:")
for pair in training_pairs:
    print(pair)
```

## Question 8: Generating CBOW Training Pairs

### 1. Implement the function generate_cbow_pairs(sentences, window_size).

### 2. Test it with the given sentences and window_size = 2.

```py
def generate_cbow_pairs(sentences, window_size):
    # Preprocessing: Build the vocabulary and word indices
    vocabulary = {}
    index = 0
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1

    training_pairs = []
    for sentence in sentences:
        words = sentence.lower().split()
        for i, target_word in enumerate(words):
            # Define the context window
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context_words = []
            for j in range(start, end):
                if i != j:
                    context_words.append(words[j])
            if context_words:
                training_pairs.append((tuple(context_words), target_word))

    return vocabulary, training_pairs


sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]

window_size = 2

vocabulary, training_pairs = generate_cbow_pairs(sentences, window_size)

print("Vocabulary:", vocabulary)
print("\nCBOW Training Pairs:")
for pair in training_pairs:
    print(f"Context: {pair[0]}, Target: {pair[1]}")
```

## Question 9: Implementing a Simple Vanilla RNN

### 1. Implement the function rnn_forward(x, Wxh, Whh, Why, bh, by, h0).

### 2. Test the function with random weights, biases, and an initial hidden state.

```py
import numpy as np

def rnn_forward(x, Wxh, Whh, Why, bh, by, h0):
    h = h0
    hs = []
    ys = []
    for t in range(len(x)):
        xt = np.array([[x[t]]])
        h = np.tanh(np.dot(Whh, h) + np.dot(Wxh, xt) + bh)
        y = np.dot(Why, h) + by
        hs.append(h)
        ys.append(y)
    return ys, hs


x = [1, 2, 3]

input_size = 1
hidden_size = 4
output_size = 1

np.random.seed(0)
Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))
h0 = np.zeros((hidden_size, 1))

ys, hs = rnn_forward(x, Wxh, Whh, Why, bh, by, h0)

print("Outputs at each time step:")
for t, y in enumerate(ys):
    print(f"Time step {t+1}: y = {y.flatten()}")
```

## Question 10 : Implementation of the self-attention mechanism using only NumPy

```py
import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(X, Wq, Wk, Wv):
    # Compute Queries (Q), Keys (K), and Values (V)
    Q = np.dot(X, Wq)
    K = np.dot(X, Wk)
    V = np.dot(X, Wv)

    d_k = Q.shape[1]
    attention_scores = np.dot(Q, K.T) / np.sqrt(d_k)

    attention_weights = softmax(attention_scores, axis=-1)

    output = np.dot(attention_weights, V)

    return output


np.random.seed(0)

X = np.random.rand(4, 3)

d = 3
dout = 2
Wq = np.random.rand(d, dout)
Wk = np.random.rand(d, dout)
Wv = np.random.rand(d, dout)

output = self_attention(X, Wq, Wk, Wv)

print("Input Matrix X:")
print(X)
print("\nWeight Matrix Wq:")
print(Wq)
print("\nWeight Matrix Wk:")
print(Wk)
print("\nWeight Matrix Wv:")
print(Wv)
print("\nSelf-Attention Output:")
print(output)
```
