FROM python:3.13-slim-bookworm

# Install uv.
RUN pip install --no-cache-dir uv

WORKDIR /code

# Install dependencies, including the spaCy model.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application.
COPY app ./app

EXPOSE 80

# Run FastAPI inside the container.
CMD ["/code/.venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]