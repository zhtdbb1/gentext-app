"""Train either Assignment 4 generator. No training occurs in the web server."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.diffusion_model import DiffusionModel, UNet
from app.energy_model import EnergyModel, langevin_sample

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['diffusion', 'energy'], required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0,
                        help='0 uses all 50000 CIFAR-10 training images')
    parser.add_argument('--langevin-steps', type=int, default=60)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.langevin_steps) < 1 or args.limit < 0 or args.batch_size > 512:
        parser.error('Positive epochs, steps; batch-size 1-512; nonnegative limit required')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    # Diffusion wrapper normalizes internally; EBM normalization is below.
    dataset = datasets.CIFAR10(ROOT / 'data', train=True, download=True,
                              transform=transforms.ToTensor())
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    model = (DiffusionModel(UNet(image_size=32, num_channels=3))
             if args.model == 'diffusion' else EnergyModel()).to(device)
    parameters = model.network.parameters() if args.model == 'diffusion' else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    checkpoint_dir = ROOT / 'models'
    checkpoint_dir.mkdir(exist_ok=True)
    results = ROOT / 'assignment4_results'
    results.mkdir(exist_ok=True)
    deployment_path = checkpoint_dir / f'cifar10_{args.model}.pth'
    resume_path = ROOT / 'data' / f'{args.model}_resume.pth'
    replay = torch.rand(512, 3, 32, 32, device=device) * 2 - 1
    history, start = [], 0
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history, start = checkpoint['history'], checkpoint['epoch']
        if checkpoint['examples'] != len(dataset):
            raise ValueError('Resume with the same --limit as the saved run, or start fresh')
        if args.model == 'energy':
            replay = checkpoint['replay']
        torch.set_rng_state(checkpoint['rng_state'].cpu())
    print(f'{args.model}: device={device}, training images={len(dataset)}', flush=True)
    for epoch in range(start, args.epochs):
        total = seen = 0
        model.train()
        for batch, (images, _) in enumerate(loader):
            images = images.to(device)
            if args.model == 'diffusion':
                loss_value = model.train_step(images, optimizer, nn.MSELoss())
            else:
                images = images * 2 - 1
                indices = torch.randperm(len(replay), device=device)[:len(images)]
                initial = replay[indices].clone()
                reset = torch.rand(len(images), 1, 1, 1, device=device) < 0.05
                initial = torch.where(reset, torch.rand_like(initial) * 2 - 1, initial)
                negative = langevin_sample(model, initial, steps=args.langevin_steps)
                replay[indices] = negative
                optimizer.zero_grad(set_to_none=True)
                real_energy, negative_energy = model(images), model(negative.detach())
                loss = (real_energy.mean() - negative_energy.mean()
                        + 0.1 * (real_energy.square().mean() + negative_energy.square().mean()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_value = loss.item()
            if not torch.isfinite(torch.tensor(loss_value)):
                raise RuntimeError('Nonfinite loss; training stopped without overwriting checkpoint')
            total += loss_value * len(images)
            seen += len(images)
            if batch % 25 == 0:
                print(f'Epoch {epoch+1}, batch {batch}/{len(loader)}: loss={loss_value:.4f}', flush=True)
        history.append({'epoch': epoch + 1, 'mean_loss': total / seen})
        model.eval()
        if args.model == 'diffusion':
            samples = model.generate(num_images=8, diffusion_steps=20)
        else:
            initial = torch.rand(8, 3, 32, 32, device=device) * 2 - 1
            samples = (langevin_sample(model, initial, steps=args.langevin_steps) + 1) / 2
        save_image(samples.cpu(), results / f'{args.model}_epoch_{epoch+1:03d}.png', nrow=4)
        metadata = {'epoch': epoch + 1, 'examples': len(dataset), 'image_size': 32,
                    'channels': 3, 'dataset': 'CIFAR10 training split',
                    'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        torch.save(metadata, deployment_path)
        torch.save({**metadata, 'optimizer': optimizer.state_dict(), 'history': history,
                    'replay': replay.detach().cpu(), 'rng_state': torch.get_rng_state()}, resume_path)
        (results / f'{args.model}_metrics.json').write_text(json.dumps({
            'model': args.model, 'device': str(device), 'training_examples': len(dataset),
            'batch_size': args.batch_size, 'langevin_steps': args.langevin_steps,
            'history': history}, indent=2) + '\n')
        print(f'Saved {deployment_path}; mean loss={total/seen:.4f}', flush=True)


if __name__ == '__main__':
    main()
