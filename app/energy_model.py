"""CIFAR-10 energy network and input-gradient Langevin sampler."""
import torch
from torch import nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(128 * 4 * 4, 1),
        )

    def forward(self, images):
        return self.layers(images).squeeze(1)


def langevin_sample(model, initial, steps=60, step_size=0.01, rng=None):
    """Approximate projected Langevin sampling for p(x) proportional to exp(-E(x))."""
    if steps < 1 or step_size <= 0:
        raise ValueError('steps and step_size must be positive')
    image = initial.detach().clone()
    # autograd.grad requests only input derivatives; weights are never updated.
    # No global requires_grad toggles, so concurrent API calls cannot interfere.
    with torch.enable_grad():
        for _ in range(steps):
            image.requires_grad_(True)
            gradient, = torch.autograd.grad(model(image).sum(), image)
            with torch.no_grad():
                noise = torch.randn(image.shape, device=image.device,
                                    dtype=image.dtype, generator=rng)
                image = (image - 0.5 * step_size * gradient
                         + step_size ** 0.5 * noise).clamp(-1, 1)
            image = image.detach()
    return image
