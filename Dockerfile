# RunPod Serverless Dockerfile for Qwen-Image-Layered
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install diffusers from GitHub for latest version (required for QwenImageLayeredPipeline)
RUN python3 -m pip install git+https://github.com/huggingface/diffusers
RUN python3 -m pip install transformers>=4.51.3 accelerate safetensors
RUN python3 -m pip install hf-transfer

# Install required dependencies for Qwen-Image-Layered
RUN python3 -m pip install pillow python-pptx

RUN python3 -m pip install runpod

RUN python3 -m pip cache purge

# Pre-download the model during build to avoid timeout during initialization
RUN python3 -c "from diffusers import QwenImageLayeredPipeline; import torch; \
    print('Downloading Qwen-Image-Layered model...'); \
    QwenImageLayeredPipeline.from_pretrained('Qwen/Qwen-Image-Layered', torch_dtype=torch.float16); \
    print('Model download complete!')"

# Copy handler
COPY handler.py /workspace/handler.py

# Point HuggingFace cache to network volume (persistent 100GB storage)
# Model downloads ONCE to volume, then ALL workers share it
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume
ENV HF_HUB_CACHE=/runpod-volume

# RunPod will execute this
CMD ["python3", "-u", "handler.py"]
