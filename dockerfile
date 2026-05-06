FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA PyTorch
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

COPY pyproject.toml ./

# Install project dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "8000"]