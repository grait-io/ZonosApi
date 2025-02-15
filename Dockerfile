FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN pip install torch torchaudio  # Explicitly install PyTorch and torchaudio

RUN pip install uv  # Install uv

RUN apt update && \
    apt install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

RUN uv pip install --system .  # Install dependencies from pyproject.toml
RUN python -c "import torch; print(torch.__version__)"  # Verify torch installation

CMD ["uvicorn", "openai_api:app", "--host", "0.0.0.0", "--reload"]
