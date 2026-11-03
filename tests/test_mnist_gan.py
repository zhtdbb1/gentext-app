import unittest
from io import BytesIO
from unittest.mock import patch

import torch
from PIL import Image
from app.mnist_gan import Generator, Discriminator
from app.digit_generator import generate_digit_png


class GANTests(unittest.TestCase):
    def test_architecture_and_optimizer_steps(self):
        torch.manual_seed(1)
        generator, discriminator = Generator(), Discriminator()
        noise = torch.randn(2, 100)
        shapes = []
        handles = [layer.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
                   for layer in generator.layers]
        fake = generator(noise)
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes[1], (2, 128, 7, 7))
        self.assertEqual(shapes[2], (2, 64, 14, 14))
        self.assertEqual(fake.shape, (2, 1, 28, 28))
        self.assertLessEqual(fake.abs().max().item(), 1)
        d_shapes = []
        hooks = [layer.register_forward_hook(lambda m, i, o: d_shapes.append(tuple(o.shape))) for layer in discriminator.layers]
        probabilities = discriminator(fake.detach())
        for hook in hooks:
            hook.remove()
        self.assertEqual(d_shapes[0], (2, 64, 14, 14))
        self.assertEqual(d_shapes[2], (2, 128, 7, 7))
        self.assertEqual(probabilities.shape, (2, 1))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        opt_g = torch.optim.Adam(generator.parameters(), lr=0.001)
        before_d = next(discriminator.parameters()).detach().clone()
        before_g = next(generator.parameters()).detach().clone()
        real = torch.randn(2, 1, 28, 28).clamp(-1, 1)
        loss_fn = torch.nn.BCELoss()
        d_loss = (loss_fn(discriminator(real), torch.ones(2, 1)) +
                  loss_fn(discriminator(fake.detach()), torch.zeros(2, 1))) / 2
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        discriminator.requires_grad_(False)
        opt_g.zero_grad()
        g_loss = loss_fn(discriminator(fake), torch.ones(2, 1))
        g_loss.backward()
        opt_g.step()
        self.assertTrue(torch.isfinite(d_loss) and torch.isfinite(g_loss))
        self.assertFalse(torch.equal(before_d, next(discriminator.parameters())))
        self.assertFalse(torch.equal(before_g, next(generator.parameters())))

    def test_png_and_seed(self):
        model = Generator().eval()
        with patch('app.digit_generator.load_generator', return_value=model):
            first = generate_digit_png(4, 7)
            self.assertEqual(first, generate_digit_png(4, 7))
            self.assertNotEqual(first, generate_digit_png(4, 8))
        image = Image.open(BytesIO(first))
        self.assertEqual(image.format, 'PNG')
        self.assertEqual(image.mode, 'L')
        self.assertEqual(image.size, (122, 32))


if __name__ == '__main__':
    unittest.main()
