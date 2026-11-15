"""Focused Assignment 4 tests; uses tiny synthetic images and temporary weights."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from app.diffusion_model import DiffusionModel, UNet
from app.energy_model import EnergyModel, langevin_sample
from app import image_generators


class GeneratorTests(unittest.TestCase):
    def test_energy_sampling_updates_only_input(self):
        model = EnergyModel()
        weights = [p.detach().clone() for p in model.parameters()]
        initial = torch.zeros(2, 3, 32, 32)
        output = langevin_sample(model, initial, steps=2)
        self.assertFalse(output.requires_grad)
        self.assertFalse(torch.equal(initial, output))
        for old, parameter in zip(weights, model.parameters()):
            self.assertTrue(torch.equal(old, parameter))
            self.assertIsNone(parameter.grad)

    def test_energy_gradient_moves_downhill(self):
        class Quadratic(torch.nn.Module):
            def forward(self, x):
                return x.square().flatten(1).sum(1) / 2
        with patch('app.energy_model.torch.randn',
                   side_effect=lambda shape, **kw: torch.zeros(shape, **{k:v for k,v in kw.items() if k != 'generator'})):
            output = langevin_sample(Quadratic(), torch.ones(2,3,32,32)*0.5,
                                     steps=2, step_size=0.1)
        self.assertTrue(torch.allclose(output, torch.ones_like(output)*0.5*0.95**2))

    def test_diffusion_training_and_ema(self):
        torch.manual_seed(42)
        model = DiffusionModel(UNet(32,3))
        optimizer = torch.optim.Adam(model.network.parameters(), lr=1e-4)
        initial = model.network.final.weight.detach().clone()
        loss = model.train_step(torch.rand(2,3,32,32), optimizer, torch.nn.MSELoss())
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(initial, model.network.final.weight))
        self.assertFalse(torch.equal(initial, model.ema_network.final.weight))
        self.assertTrue(all(p.grad is None for p in model.ema_network.parameters()))

    def test_checkpoints_png_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'models').mkdir()
            for kind, model in [('diffusion', DiffusionModel(UNet(32,3))), ('energy', EnergyModel())]:
                torch.save({'state_dict':model.state_dict()},root/'models'/f'cifar10_{kind}.pth')
            with patch.object(image_generators,'ROOT',root):
                image_generators.load_generator.cache_clear()
                try:
                    for kind in ['diffusion','energy']:
                        first = image_generators.image_grid_png(kind,2,2,42)
                        self.assertEqual(first,image_generators.image_grid_png(kind,2,2,42))
                        image = Image.open(BytesIO(first));image.load()
                        self.assertEqual(image.mode,'RGB')
                        self.assertEqual(image.size,(70,36))
                finally:
                    image_generators.load_generator.cache_clear()


if __name__ == '__main__':
    unittest.main()
