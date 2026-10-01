import spacy


class EmbeddingModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def get_embedding(self, word: str) -> dict:
        word = word.strip()

        if not word:
            raise ValueError("Please enter a word.")

        doc = self.nlp(word)

        if len(doc) != 1 or not doc[0].is_alpha:
            raise ValueError("Please enter one alphabetic word.")

        if not doc[0].has_vector:
            raise ValueError("The model has no embedding for this word.")

        return {
            "word": word,
            "model": "en_core_web_lg",
            "dimensions": int(doc.vector.size),
            "embedding": doc.vector.tolist(),
        }