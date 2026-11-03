# Assignment 3: MNIST GAN

**Haotian Zhu - HZ3102**

This implementation adds an unconditional MNIST digit generator to the existing FastAPI application. The original text generation, embeddings, image classification, and LSTM routes remain available.

## Architecture

| Generator layer | Output per image |
|---|---|
| Noise | 100 |
| Linear and reshape | 128 x 7 x 7 |
| ConvTranspose2d(128, 64, 4, 2, 1), BatchNorm, ReLU | 64 x 14 x 14 |
| ConvTranspose2d(64, 1, 4, 2, 1), Tanh | 1 x 28 x 28 |

| Discriminator layer | Output per image |
|---|---|
| Input | 1 x 28 x 28 |
| Conv2d(1, 64, 4, 2, 1), LeakyReLU(0.2) | 64 x 14 x 14 |
| Conv2d(64, 128, 4, 2, 1), BatchNorm, LeakyReLU(0.2) | 128 x 7 x 7 |
| Flatten, Linear(6272, 1), Sigmoid | One real/fake probability |

## Training

From the project root:

```bash
uv sync --frozen --no-dev --no-install-project
uv run --no-sync python train_gan.py --epochs 10
```

MNIST is downloaded automatically into the ignored `data/` directory. All 60,000 training images are normalized from [0, 1] to [-1, 1] to match the generator's Tanh. Labels are unused. Each batch updates the discriminator once using real and detached generated images, followed by one generator update targeting real labels. The loss is binary cross entropy; both optimizers use Adam, learning rate 0.0002, betas (0.5, 0.999). Discriminator parameters are frozen during the generator update; gradients still flow through the discriminator into the generator. Both adversarial updates use training-mode BatchNorm, so the generator learns against the same discriminator behavior. Inference uses evaluation mode.

Each completed epoch saves:

- `models/mnist_generator.pth`: generator weights and metadata for deployment.
- `gan_results/epoch_NNN.png`: a grid from 64 fixed noise vectors.
- `gan_results/training_metrics.json`: actual losses, configuration, and epoch durations.
- `data/mnist_gan_resume.pth`: training state, excluded from Git and Docker.

Continue to a total of 10 epochs from the last completed epoch:

```bash
uv run --no-sync python train_gan.py --resume --epochs 10
```

The supplied checkpoint was trained for **3 complete epochs on all 60,000 MNIST training images**, using CPU, batch size 128, and seed 42. The completed epoch count is also recorded in the checkpoint and metrics. Saved grids contain recognizable digits alongside rough or ambiguous samples; this is a trained classroom demonstration rather than a uniformly high-quality digit generator. More epochs may improve images, but GAN losses alone do not establish image quality. Inspect the saved grids for legibility and variety. This is an unconditional GAN: it does not accept a requested digit class and does not guarantee that every output is recognizable.

## Run and query the API

```bash
docker build -t gentext-app .
docker run --rm -p 8000:80 gentext-app
```

Open http://127.0.0.1:8000/docs and use **POST /generate_digits** with:

```json
{"count": 16, "seed": 42}
```

The response is an `image/png` grid. Each digit is 28 x 28 pixels, with padding between tiles. `count` must be 1-64 and `seed` must be 0-2147483647. The same seed and checkpoint reproduce the same grid. Different seeds produce different samples. Training occurs offline, never inside the request handler. Missing weights return HTTP 503 with a training instruction; invalid parameters return HTTP 422.

To save an image from Terminal:

```bash
curl --fail -X POST http://127.0.0.1:8000/generate_digits \
  -H 'Content-Type: application/json' \
  -d '{"count":16,"seed":42}' \
  --output generated_digits.png
```

Linux installs use the official CPU-only PyTorch index, selected with platform markers in `pyproject.toml`; macOS continues to use PyPI. This avoids NVIDIA CUDA downloads for the CPU deployment. The lock file was regenerated with uv. See [uv PyTorch configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).

The Dockerfile copies `app/` and `models/`, including the trained GAN. Rebuild after updating weights. Ordinary starts can reuse the existing image.

## Validation

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

The focused tests check spatial dimensions, finite losses, actual parameter updates, PNG decoding, and seed reproducibility. Existing API integration and Docker checks are recorded in `gan_results/verification.json` when performed.

## Data source

MNIST is loaded through `torchvision.datasets.MNIST`, which downloads and verifies its original compressed files. The loader's fallback mirror is https://ossci-datasets.s3.amazonaws.com/mnist/. Training uses the training split only; the downloaded test split is not used to optimize the GAN.
