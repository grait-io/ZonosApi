FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN pip install torch torchaudio  # Explizit PyTorch und torchaudio installieren

RUN pip install uv  # uv installieren

RUN apt update && \
    apt install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

RUN uv pip install --system -e . && uv pip install --system -e .[compile] # Install base and optional dependencies
RUN python -c "import torch; print(torch.__version__)"  # Verify torch installation

# Debugging Befehle:
# RUN which python
# RUN python -c "import sys; print(sys.path)"
# RUN pip list

# CMD ["uvicorn", "openai_api:app", "--host", "0.0.0.0", "--reload"]
CMD ["/bin/bash", "-c", "source /opt/conda/bin/activate base && echo 'PATH:' $PATH && echo 'PYTHONPATH:' $PYTHONPATH && uvicorn openai_api:app --host 0.0.0.0 --reload"]
