import copy
import math

import torch
from torch import nn
from torch.nn import functional as F



class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)         

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.proj(x)
        # x = self.norm(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        image_size=32,
        num_channels=3,
        embedding_dim=32,
    ):
        super().__init__()

        self.image_size = image_size
        self.num_channels = num_channels

        self.initial = nn.Conv2d(
            num_channels, 32, kernel_size=1
        )

        self.embedding = SinusoidalEmbedding(
            num_frequencies=embedding_dim // 2
        )

        self.down1 = DownBlock(
            width=32,
            block_depth=2,
            in_channels=32 + embedding_dim,
        )
        self.down2 = DownBlock(
            width=64,
            block_depth=2,
            in_channels=32,
        )
        self.down3 = DownBlock(
            width=96,
            block_depth=2,
            in_channels=64,
        )

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(
            width=96,
            block_depth=2,
            in_channels=128,
        )
        self.up2 = UpBlock(
            width=64,
            block_depth=2,
            in_channels=96,
        )
        self.up3 = UpBlock(
            width=32,
            block_depth=2,
            in_channels=64,
        )

        self.final = nn.Conv2d(
            32, num_channels, kernel_size=1
        )

        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, noisy_images, noise_variances):
        skips = []

        x = self.initial(noisy_images)

        noise_embedding = self.embedding(noise_variances)
        noise_embedding = noise_embedding.permute(
            0, 3, 1, 2
        )

        noise_embedding = F.interpolate(
            noise_embedding,
            size=noisy_images.shape[-2:],
            mode="nearest",
        )

        x = torch.cat([x, noise_embedding], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)

def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95,
):
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)

    angles = start_angle + diffusion_times * (
        end_angle - start_angle
    )

    return torch.sin(angles), torch.cos(angles)


class DiffusionModel(nn.Module):
    def __init__(self, network, ema_decay=0.999):
        super().__init__()

        self.network = network
        self.ema_network = copy.deepcopy(network)
        self.ema_network.requires_grad_(False)
        self.ema_network.eval()
        self.ema_decay = ema_decay

        # Training data starts in [0, 1].
        # These values map it to [-1, 1].
        self.register_buffer(
            "normalizer_mean",
            torch.tensor(0.5),
        )
        self.register_buffer(
            "normalizer_std",
            torch.tensor(0.5),
        )

    def train_step(self, images, optimizer, criterion):
        self.network.train()

        clean_images = (
            images - self.normalizer_mean
        ) / self.normalizer_std

        noise = torch.randn_like(clean_images)

        times = torch.rand(
            images.size(0),
            1,
            1,
            1,
            device=images.device,
        )

        noise_rates, signal_rates = (
            offset_cosine_diffusion_schedule(times)
        )

        noisy_images = (
            signal_rates * clean_images
            + noise_rates * noise
        )

        predicted_noise = self.network(
            noisy_images,
            noise_rates.square(),
        )

        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Maintain an exponential moving average for sampling.
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self.ema_network.parameters(),
                self.network.parameters(),
            ):
                ema_parameter.mul_(self.ema_decay)
                ema_parameter.add_(
                    parameter,
                    alpha=1 - self.ema_decay,
                )

            for ema_buffer, buffer in zip(
                self.ema_network.buffers(),
                self.network.buffers(),
            ):
                ema_buffer.copy_(buffer)

        return loss.item()

    @torch.no_grad()
    def generate(self, num_images=10, diffusion_steps=100, initial_noise=None):
        if num_images < 1 or diffusion_steps < 1:
            raise ValueError(
                "num_images and diffusion_steps must be positive."
            )

        self.ema_network.eval()

        device = next(self.network.parameters()).device
        size = self.network.image_size

        current_images = torch.randn(
            num_images,
            self.network.num_channels,
            size,
            size,
            device=device,
        )

        if initial_noise is not None:
            expected = (num_images, self.network.num_channels, size, size)
            if tuple(initial_noise.shape) != expected:
                raise ValueError(f"initial_noise must have shape {expected}")
            current_images = initial_noise.detach().to(device).clone()

        for step in range(diffusion_steps):
            times = torch.full(
                (num_images, 1, 1, 1),
                1 - step / diffusion_steps,
                device=device,
            )

            noise_rates, signal_rates = (
                offset_cosine_diffusion_schedule(times)
            )

            predicted_noise = self.ema_network(
                current_images,
                noise_rates.square(),
            )

            predicted_images = (
                current_images
                - noise_rates * predicted_noise
            ) / signal_rates

            next_times = (
                times - 1 / diffusion_steps
            ).clamp_min(0)

            next_noise_rates, next_signal_rates = (
                offset_cosine_diffusion_schedule(next_times)
            )

            current_images = (
                next_signal_rates * predicted_images
                + next_noise_rates * predicted_noise
            )

        return (
            predicted_images * self.normalizer_std
            + self.normalizer_mean
        ).clamp(0, 1)
