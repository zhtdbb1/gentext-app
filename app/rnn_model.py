import re
from pathlib import Path

import torch
from torch import nn


def tokenize(text):
    text = text.lower().replace("’", "'")

    return re.findall(
        r"\w+(?:'\w+)*|[.,!?;:]",
        text,
    )


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        logits = self.fc(x)

        return logits, hidden


class RNNTextGenerator:
    def __init__(self):
        checkpoint_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "lstm_text.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Train the LSTM first: "
                "uv run python train_lstm.py"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        self.words = checkpoint["vocabulary"]

        self.vocab = {
            word: index
            for index, word in enumerate(self.words)
        }

        self.model = LSTMModel(
            **checkpoint["config"]
        )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.model.eval()

    def generate_text(
        self,
        seed_text,
        length=50,
        temperature=1.0,
    ):
        words = tokenize(seed_text)

        if not words:
            raise ValueError(
                "Please provide a nonempty seed."
            )

        if not any(word in self.vocab for word in words):
            raise ValueError(
                "The seed contains no known vocabulary. "
                "Try 'the' or 'he'."
            )

        input_ids = [
            self.vocab.get(word, 0)
            for word in words
        ]

        inputs = torch.tensor(
            [input_ids],
            dtype=torch.long,
        )

        with torch.inference_mode():
            # Process the complete seed once.
            outputs, hidden = self.model(inputs)

            for step in range(length):
                logits = outputs[0, -1] / temperature

                # Avoid generating the unknown-word marker.
                logits[self.vocab["<UNK>"]] = -float("inf")

                probabilities = torch.softmax(
                    logits,
                    dim=-1,
                )

                next_id = torch.multinomial(
                    probabilities,
                    num_samples=1,
                ).item()

                words.append(self.words[next_id])

                if step + 1 < length:
                    # Carry the hidden state and feed only
                    # the newly generated token.
                    inputs = torch.tensor(
                        [[next_id]],
                        dtype=torch.long,
                    )

                    outputs, hidden = self.model(
                        inputs,
                        hidden,
                    )

        text = " ".join(words)

        return re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            text,
        )