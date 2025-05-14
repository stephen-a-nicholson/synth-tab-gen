FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc && \
    . ~/.bashrc

ENV PATH="/root/.cargo/bin:${PATH}"

COPY pyproject.toml .

RUN pip install -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "synth_tab_gen_backend.api:app", "--host", "0.0.0.0", "--port", "8000"]