FROM python:3.13-slim

WORKDIR /app

# Reproducible lockfile install (CPU-only, no CUDA/HF needed).
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Copy source.
COPY . .

# Lint gate (non-blocking in container; CI is the source of truth).
RUN python -m ruff check . || true

CMD ["python", "examples/run_demo.py"]
