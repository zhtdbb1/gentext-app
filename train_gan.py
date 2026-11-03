"""Train the exact Assignment 3 GAN on MNIST and save reproducible evidence."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from app.mnist_gan import Generator, Discriminator

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1:
        parser.error('epochs, batch-size, and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(ROOT / 'data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    generator, discriminator = Generator().to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    model_dir, report_dir = ROOT / 'models', ROOT / 'gan_results'
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    resume_path = ROOT / 'data' / 'mnist_gan_resume.pth'
    history = []
    first_epoch = 0
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        opt_g.load_state_dict(checkpoint['opt_g'])
        opt_d.load_state_dict(checkpoint['opt_d'])
        history = checkpoint['history']
        first_epoch = checkpoint['epoch']
        torch.set_rng_state(checkpoint['rng_state'].cpu())
    fixed_noise = torch.randn(64, 100, generator=torch.Generator().manual_seed(args.seed)).to(device)
    print(f'Device: {device}; MNIST training images: {len(dataset)}', flush=True)
    for epoch in range(first_epoch, args.epochs):
        started = time.monotonic()
        generator.train()
        discriminator.train()
        sum_g = sum_d = seen = 0
        for batch, (real, _) in enumerate(loader):
            real = real.to(device)
            n = real.size(0)
            ones, zeros = torch.ones(n, 1, device=device), torch.zeros(n, 1, device=device)
            opt_d.zero_grad(set_to_none=True)
            fake = generator(torch.randn(n, 100, device=device))
            d_loss = (criterion(discriminator(real), ones) +
                      criterion(discriminator(fake.detach()), zeros)) / 2
            d_loss.backward()
            opt_d.step()
            # Freeze D weights, but retain training-mode BatchNorm for the same
            # discriminator function in both adversarial updates.
            discriminator.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake), ones)
            g_loss.backward()
            opt_g.step()
            discriminator.train().requires_grad_(True)
            sum_g += g_loss.item() * n
            sum_d += d_loss.item() * n
            seen += n
            if batch % 100 == 0:
                print(f'Epoch {epoch+1}, batch {batch}/{len(loader)}: D={d_loss.item():.4f}, G={g_loss.item():.4f}', flush=True)
        history.append({'epoch': epoch + 1, 'generator_loss': sum_g / seen,
                        'discriminator_loss': sum_d / seen, 'seconds': time.monotonic() - started})
        generator.eval()
        with torch.inference_mode():
            save_image((generator(fixed_noise).cpu() + 1) / 2,
                       report_dir / f'epoch_{epoch+1:03d}.png', nrow=8)
        torch.save({'generator_state_dict': {k: v.cpu() for k, v in generator.state_dict().items()},
                    'epoch': epoch + 1, 'latent_dim': 100, 'seed': args.seed,
                    'dataset': 'MNIST training split (60000 images)'}, model_dir / 'mnist_generator.pth')
        torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(),
                    'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(), 'history': history,
                    'epoch': epoch + 1, 'rng_state': torch.get_rng_state()}, resume_path)
        (report_dir / 'training_metrics.json').write_text(json.dumps({
            'dataset': 'MNIST', 'training_images': len(dataset), 'seed': args.seed,
            'batch_size': args.batch_size, 'optimizer': 'Adam', 'learning_rate': 0.0002,
            'betas': [0.5, 0.999], 'loss': 'binary cross entropy', 'device': str(device),
            'history': history}, indent=2) + '\n')
        print(f'Completed epoch {epoch+1}: {history[-1]}', flush=True)


if __name__ == '__main__':
    main()
