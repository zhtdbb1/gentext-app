import argparse
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from app.rnn_model import LSTMModel, tokenize


ROOT = Path(__file__).resolve().parent
SEQ_LEN = 30


class TextDataset(Dataset):
    def __init__(self, encoded):
        self.data = torch.tensor(
            encoded,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.data) - SEQ_LEN

    def __getitem__(self, index):
        inputs = self.data[
            index:index + SEQ_LEN
        ]

        targets = self.data[
            index + 1:index + SEQ_LEN + 1
        ]

        return inputs, targets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20000,
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.max_tokens <= SEQ_LEN:
        raise ValueError(
            "Use positive epochs and more than "
            "30 training tokens."
        )

    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    text = (ROOT / "data" / "corpus.txt").read_text(
        encoding="utf-8"
    )

    # Remove Gutenberg's surrounding header/footer
    # when its standard markers are present.
    match = re.search(
        r"\*\*\* START OF .*?\*\*\*"
        r"(.*?)"
        r"(?:\*\*\* END OF|\Z)",
        text,
        flags=re.DOTALL,
    )

    if match:
        text = match.group(1)

    tokens = tokenize(text)[:args.max_tokens]

    if len(tokens) <= SEQ_LEN:
        raise ValueError(
            "The corpus needs more than 30 tokens."
        )

    counts = Counter(tokens)

    # At most 10,000 entries, including <UNK>.
    vocabulary = ["<UNK>"] + [
        word
        for word, _ in counts.most_common(9999)
    ]

    vocab = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    encoded = [
        vocab.get(word, 0)
        for word in tokens
    ]

    dataset = TextDataset(encoded)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    config = {
        "vocab_size": len(vocabulary),
        "embedding_dim": 100,
        "hidden_dim": 128,
    }

    model = LSTMModel(**config).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print(f"Device: {device}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Vocabulary size: {len(vocabulary):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_tokens = 0

        for batch, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs, _ = model(inputs)

            loss = criterion(
                outputs.reshape(-1, len(vocabulary)),
                targets.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            count = targets.numel()
            total_loss += loss.item() * count
            total_tokens += count

            if batch % 100 == 0:
                print(
                    f"Epoch {epoch}, batch {batch}: "
                    f"loss={loss.item():.4f}"
                )

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"average loss="
            f"{total_loss / total_tokens:.4f}"
        )

    checkpoint = {
        "config": config,
        "vocabulary": vocabulary,
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
    }

    output_path = ROOT / "models" / "lstm_text.pth"
    output_path.parent.mkdir(exist_ok=True)

    torch.save(checkpoint, output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()