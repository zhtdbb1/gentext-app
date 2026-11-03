"""Load trained weights once and return a PNG grid of generated digits."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid
from app.mnist_gan import Generator

CHECKPOINT = Path(__file__).resolve().parents[1] / 'models' / 'mnist_generator.pth'


@lru_cache(maxsize=1)
def load_generator():
    if not CHECKPOINT.exists():
        raise FileNotFoundError('Train the GAN first: uv run python train_gan.py')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = Generator()
    model.load_state_dict(checkpoint['generator_state_dict'])
    return model.eval()


def generate_digit_png(count=16, seed=42):
    model = load_generator()
    # Local RNG avoids interference between concurrent requests.
    rng = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(count, 100, generator=rng)
    with torch.inference_mode():
        images = ((model(noise) + 1) / 2).clamp(0, 1)
    grid = make_grid(images, nrow=min(8, count), padding=2)
    pixels = grid[0].mul(255).round().to(torch.uint8).numpy()
    output = BytesIO()
    Image.fromarray(pixels, mode='L').save(output, format='PNG')
    return output.getvalue()
