# Stage 1: Build CUDA extensions e Dependências (Blindado)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA=1 \
    MAX_JOBS=1 \
    PIP_NO_CACHE_DIR=1

# Ajuste: Adicionadas bibliotecas X11 (libxrender1 libxext6 libxi6) para o motor do Blender (bpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    libgl1 libglib2.0-0 libgomp1 \
    ninja-build build-essential \
    libosmesa6-dev libglfw3-dev libgles2-mesa-dev \
    libxrender1 libxext6 libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Criar VENV herdando o PyTorch (CUDA 12.1) já instalado na imagem base
RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CORREÇÃO CRÍTICA: Apontando para o repositório oficial da v2.1
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /Hunyuan3D-2.1
WORKDIR /Hunyuan3D-2.1

RUN pip install --upgrade pip setuptools wheel

# Restrições de compatibilidade
RUN pip install --no-cache-dir "numpy<2" "diffusers<0.31"

# Cirurgia no requirements.txt oficial
RUN tr -d '\r' < requirements.txt > req_clean.txt && \
    sed -i -E '/^(torch|torchvision|torchaudio|xformers|flash-attn|flash_attn|triton)/Id' req_clean.txt && \
    sed -i 's/bpy==4\.0/bpy/g' req_clean.txt

# Instalação segura (preferir binários, sem compilações pesadas)
RUN pip install --no-cache-dir --prefer-binary --retries 5 --timeout 120 -r req_clean.txt

# Instalar manualmente alguns extras úteis (opcional)
RUN pip install --no-cache-dir \
    accelerate \
    transformers \
    einops \
    omegaconf \
    trimesh \
    pygltflib \
    Pillow \
    huggingface_hub \
    tqdm \
    matplotlib \
    scipy \
    opencv-python-headless

# Requirements customizados (runpod, boto3)
COPY requirements.txt /tmp/custom_reqs.txt
RUN pip install --no-cache-dir -r /tmp/custom_reqs.txt

# Compilar extensões CUDA usando os comandos oficiais da documentação v2.1
RUN cd hy3dpaint/custom_rasterizer && pip install --no-build-isolation -e .
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh

# Limpeza
RUN pip cache purge && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime image (apenas execução)
# ─────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    VOLUME_PATH=/runpod-volume \
    MAX_IMAGE_SIZE_MB=5 \
    MAX_MESH_SIZE_MB=100 \
    PRELOAD_MODELS=false

# Instalar APENAS bibliotecas de sistema necessárias para execução
# Ajuste: Adicionadas bibliotecas X11 (libxrender1 libxext6 libxi6) para o runtime do Blender
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 wget curl \
    libxrender1 libxext6 libxi6 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copiar o VENV completo (com todas as dependências e binários compilados)
COPY --from=builder /opt/venv /opt/venv

# Copiar código-fonte do Hunyuan3D-2.1
COPY --from=builder /Hunyuan3D-2.1 /Hunyuan3D-2.1

# Criar diretório do volume persistente
RUN mkdir -p /runpod-volume

# Copiar handler para dentro do diretório do modelo
COPY handler.py /Hunyuan3D-2.1/handler.py

# Definir diretório de trabalho
WORKDIR /Hunyuan3D-2.1

# Comando de inicialização
CMD ["python", "-u", "/Hunyuan3D-2.1/handler.py"]