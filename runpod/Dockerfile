# ─────────────────────────────────────────────────────────────
#  Base: RunPod PyTorch + CUDA 12.1 + Python 3.10
# ─────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

LABEL maintainer="mash-stl"
LABEL description="Hunyuan3D-2 serverless endpoint para RunPod (alta qualidade)"

# Evita prompts interativos do apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Arquiteturas CUDA para compilar custom rasterizer sem GPU no build
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ENV FORCE_CUDA=1

# ─────────────────────────────────────────────────────────────
#  Dependências de sistema
# ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────
#  Clona Hunyuan3D-2
# ─────────────────────────────────────────────────────────────
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /Hunyuan3D-2
WORKDIR /Hunyuan3D-2

# ─────────────────────────────────────────────────────────────
#  Instala PyTorch (já vem na base, mas garante versão correta)
# ─────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ─────────────────────────────────────────────────────────────
#  Instala dependências do Hunyuan3D-2
# ─────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# ─────────────────────────────────────────────────────────────
#  Compila custom rasterizer e differentiable renderer (textura)
# ─────────────────────────────────────────────────────────────
RUN cd hy3dgen/texgen/custom_rasterizer && \
    python setup.py install && \
    cd /Hunyuan3D-2

RUN cd hy3dgen/texgen/differentiable_renderer && \
    python setup.py install && \
    cd /Hunyuan3D-2

# ─────────────────────────────────────────────────────────────
#  Instala SDK do RunPod + dependências do handler
# ─────────────────────────────────────────────────────────────
COPY requirements.txt /tmp/runpod_requirements.txt
RUN pip install --no-cache-dir -r /tmp/runpod_requirements.txt

# ─────────────────────────────────────────────────────────────
#  Pré-download dos pesos (evita cold start longo)
#  Shape model (DiT) + Texture model (Paint) + Delight
# ─────────────────────────────────────────────────────────────
RUN python -c "\
    from huggingface_hub import snapshot_download; \
    snapshot_download( \
    repo_id='tencent/Hunyuan3D-2', \
    local_dir='/Hunyuan3D-2/weights', \
    ignore_patterns=['*.md', '*.txt', '*.git*'] \
    ); \
    print('Pesos Hunyuan3D-2 baixados com sucesso.') \
    "

# ─────────────────────────────────────────────────────────────
#  Copia o handler
# ─────────────────────────────────────────────────────────────
COPY handler.py /handler.py

# ─────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────
CMD ["python", "-u", "/handler.py"]
