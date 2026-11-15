"""CPU inference services for the two CIFAR-10 generators."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid
from app.diffusion_model import DiffusionModel, UNet
from app.energy_model import EnergyModel, langevin_sample

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=2)
def load_generator(kind):
    if kind not in {'diffusion', 'energy'}:
        raise ValueError('Unknown generator')
    path = ROOT / 'models' / f'cifar10_{kind}.pth'
    if not path.exists():
        raise FileNotFoundError(f'Train first: uv run --no-sync python train_image_generators.py --model {kind}')
    model = (DiffusionModel(UNet(image_size=32, num_channels=3))
             if kind == 'diffusion' else EnergyModel())
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().requires_grad_(False)
    return model


def image_grid_png(kind, count=4, steps=50, seed=42):
    model = load_generator(kind)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    if kind == 'diffusion':
        noise = torch.randn(count, 3, 32, 32, generator=rng)
        samples = model.generate(num_images=count, diffusion_steps=steps,
                                 initial_noise=noise)
    else:
        initial = torch.rand(count, 3, 32, 32, generator=rng) * 2 - 1
        # Do not enclose this branch in torch.no_grad()/inference_mode().
        samples = (langevin_sample(model, initial, steps=steps, rng=rng) + 1) / 2
    grid = make_grid(samples.detach().cpu().clamp(0, 1), nrow=min(4, count))
    pixels = grid.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    output = BytesIO()
    Image.fromarray(pixels).save(output, format='PNG')
    return output.getvalue()
