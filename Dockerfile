FROM python:3.12-slim

# Install uv, which reads this project's locked dependencies from uv.lock.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first so Docker can reuse this layer when app code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application after installing its dependencies.
COPY app ./app

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
