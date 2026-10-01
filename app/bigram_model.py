from collections import defaultdict, Counter
import random
import re


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        """
        Initialize the bigram model using a corpus.

        Args:
            corpus (list[str]): List of text strings.
            frequency_threshold (int | None):
                Minimum word frequency required to keep a word.
        """
        self.frequency_threshold = frequency_threshold

        # Combine all corpus strings into one text
        text = " ".join(corpus)

        # Build vocabulary and bigram probabilities
        self.vocab, self.bigram_probs = self.analyze_bigrams(text)

    def simple_tokenizer(self, text):
        """
        Convert text to lowercase and split it into words.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())

        # If no threshold is given, keep all tokens
        if not self.frequency_threshold:
            return tokens

        # Count word frequencies
        word_counts = Counter(tokens)

        # Keep only words that meet the frequency threshold
        filtered_tokens = [
            token
            for token in tokens
            if word_counts[token] >= self.frequency_threshold
        ]

        return filtered_tokens

    def analyze_bigrams(self, text):
        """
        Analyze text and calculate bigram probabilities.
        """
        words = self.simple_tokenizer(text)

        # Create bigrams
        bigrams = list(zip(words[:-1], words[1:]))

        # Count bigram and unigram frequencies
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        # Calculate bigram probabilities
        bigram_probs = defaultdict(dict)

        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = (
                count / unigram_counts[word1]
            )

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, num_words=20):
        """
        Generate text using the bigram probability model.
        """
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)

            # Stop if the current word has no possible next word
            if not next_words:
                break

            # Randomly choose the next word using bigram probabilities
            next_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values())
            )[0]

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)