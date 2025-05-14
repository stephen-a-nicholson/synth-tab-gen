FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy pyproject.toml
COPY pyproject.toml .

# Use uv to install the project and dependencies
RUN uv pip install --system -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "synth_tab_gen_backend.api:app", "--host", "0.0.0.0", "--port", "8000"]