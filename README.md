# FastAPI Text Generation and Word Embeddings

This project extends the Module 3 FastAPI application with the spaCy word-embedding functionality demonstrated in Module 2.

It provides endpoints for:
- Generating text using a bigram model.
- Returning a word embedding using spaCy's `en_core_web_lg` model.

## Requirements

For local execution:
- Python 3.13
- uv

For container execution:
- Docker Desktop, or Docker Engine

An internet connection is required during initial dependency installation and Docker builds.

## Project Structure

```text
gentext-app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── bigram_model.py
│   └── embedding_model.py
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Run Locally

From the project root, install the dependencies:

```bash
uv sync
```

The dependencies include spaCy and the `en_core_web_lg` model. The first installation may take several minutes.

Start the API:

```bash
uv run uvicorn app.main:app --reload
```

Open:
- API root: http://127.0.0.1:8000
- Interactive documentation: http://127.0.0.1:8000/docs

Press `Ctrl+C` to stop the server.

## Run with Docker

Start Docker, then run these commands from the project root:

```bash
docker build -t gentext-app .
```

After the build succeeds:

```bash
docker run --rm -p 8000:80 gentext-app
```

Open the interactive documentation:

http://127.0.0.1:8000/docs

The container listens on port `80`, which is mapped to port `8000` on your computer. Stop any local server using port `8000` before starting the container.

The Docker build installs the dependencies and spaCy model automatically. Python and uv do not need to be installed on the host when using Docker.

Press `Ctrl+C` to stop the container. The `--rm` option removes the stopped container.

## API Endpoints

### GET /

Returns a basic response to confirm that the server is reachable.

Example:

```bash
curl http://127.0.0.1:8000/
```

Response:

```json
{
  "Hello": "World"
}
```

### POST /generate

Generates text using the existing bigram model.

Request fields:
- `start_word`: The starting word.
- `length`: The requested generation length.

Example request:

```json
{
  "start_word": "the",
  "length": 10
}
```

Test with:

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"start_word":"the","length":10}'
```

The response contains a `generated_text` field. The generated text depends on the training corpus and the model's sampling behavior.

### POST /embedding

Returns the complete 300-dimensional embedding for one alphabetic word using spaCy's `en_core_web_lg` model.

Example request:

```json
{
  "word": "apple"
}
```

Test with:

```bash
curl -X POST "http://127.0.0.1:8000/embedding" \
  -H "Content-Type: application/json" \
  -d '{"word":"apple"}'
```

The JSON response contains:

| Field | Description |
| --- | --- |
| `word` | The query word with surrounding whitespace removed |
| `model` | `en_core_web_lg` |
| `dimensions` | The number of values in the embedding, normally `300` |
| `embedding` | The complete list of embedding values |

The model is loaded once when the application starts. The embedding is obtained using `nlp(word).vector` and converted to a Python list for JSON serialization.

### Input Validation

The embedding endpoint returns HTTP `422` when:
- The input is empty or contains only whitespace.
- The input is not one alphabetic token.
- The model has no stored embedding for the word.
- The required `word` field is missing or has an invalid type.

Example invalid request:

```json
{
  "word": "two words"
}
```

## Testing through the Interactive Documentation

1. Start the server locally or in Docker.
2. Open http://127.0.0.1:8000/docs.
3. Expand `POST /embedding`.
4. Click **Try it out**.
5. Enter `{"word": "apple"}`.
6. Click **Execute**.
7. Verify that the response status is `200`, the dimensions are `300`, and the embedding contains 300 numbers.
8. Repeat with another word, such as `car`.
9. Test an empty string and confirm that the response status is `422`.
10. Test `POST /generate` to confirm that text generation still works.

Opening `/generate` or `/embedding` directly in the browser's address bar sends a GET request and returns `405 Method Not Allowed`. Use the interactive documentation or the POST commands above.

## Implementation

- `app/main.py` defines the API routes and request models.
- `app/bigram_model.py` implements the existing bigram text generator.
- `app/embedding_model.py` loads spaCy, validates input, and returns embeddings.
- `pyproject.toml` and `uv.lock` record the project dependencies.
- `Dockerfile` packages the application and dependencies for container execution.

The word-embedding endpoint uses a pretrained model; it does not train a new model. Words without stored vectors are rejected with an explanatory error.

## Course Context

Created for Assignment 1 of Applied Generative AI, extending the Module 3 API with the word-embedding functionality from Module 2.


## Assignment 3: MNIST GAN

See [Assignment 3 setup, training, architecture, and API instructions](ASSIGNMENT3.md). POST `/generate_digits` returns a PNG grid of generated handwritten digits.


## Assignment 4: CIFAR-10 diffusion and energy generators

Both models generate unconditional 32x32 RGB CIFAR-10-style image grids.

- `POST /generate_diffusion`: reverse diffusion using the EMA UNet.
- `POST /generate_energy`: projected Langevin sampling using input gradients.
- JSON example: `{"count":4,"steps":50,"seed":42}`. Responses are PNG images.
- `count` is 1-8, `steps` is 1-200, and `seed` is 0-2147483647.
- Missing checkpoints return 503; invalid inputs return 422.

Install dependencies with `uv sync --frozen --no-dev --no-install-project`.
Train each model from the project root:

```bash
uv run --no-sync python train_image_generators.py --model diffusion --epochs 10 --batch-size 32
uv run --no-sync python train_image_generators.py --model energy --epochs 10 --batch-size 32 --langevin-steps 60
```

Without `--limit`, all 50,000 CIFAR-10 training images are used; labels are ignored.
Add `--resume` to continue from the last completed epoch, using the same data limit and sampling settings.
Training uses Adam (learning rate 0.0001). Diffusion minimizes noise-prediction MSE;
the EBM lowers real-image energy relative to detached negatives, with squared-energy regularization.
EBM sampling differentiates energy with respect to image pixels and does not update model weights.
The sampler uses 60 steps in the full training configuration; finite-step projected sampling is approximate.

Checkpoints are `models/cifar10_diffusion.pth` and `models/cifar10_energy.pth`.
Epoch sample grids and actual training metrics are in `assignment4_results/`.
Optimizer/resume states and the CIFAR-10 dataset stay in ignored `data/`.
Training metadata and losses should be consulted for the actual completed run;
an in-progress run must not be reported as ten completed epochs.

Run focused and existing tests with:

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

Start Docker Desktop, then build and run:

```bash
docker build -t gentext-app .
docker run --rm -p 8000:80 gentext-app
```

Open http://127.0.0.1:8000/docs and query the new endpoints. To save images:

```bash
curl --fail -X POST http://127.0.0.1:8000/generate_diffusion -H 'Content-Type: application/json' -d '{"count":4,"steps":50,"seed":42}' --output diffusion_samples.png
curl --fail -X POST http://127.0.0.1:8000/generate_energy -H 'Content-Type: application/json' -d '{"count":4,"steps":50,"seed":42}' --output energy_samples.png
```

The Dockerfile includes both `app/` and `models/`. Rebuild after changing weights.
Models load once per server process, so restart after replacing a local checkpoint.
The same seed and checkpoint reproduce the same image grid. These generators are
unconditional and do not support choosing a requested CIFAR-10 class.

Full CPU training is slow, and ten epochs does not guarantee recognizable images.
Inspect saved sample grids for quality and diversity; successful HTTP responses alone
only establish that deployment works. Small `--limit 128 --epochs 1` runs are smoke
checks and must not be presented as full-data trained results.
