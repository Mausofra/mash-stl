# Multi-stage Dockerfile para Hunyuan3D-2.1 RunPod Worker
# ─────────────────────────────────────────────────────────────
# Stage 1: Build CUDA extensions e Dependências
# ─────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA=1

# Instalar dependências de sistema para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 libgomp1 ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/*

# Criar e ativar Virtual Environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clonar repositório oficial (branch main)
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /Hunyuan3D-2

# Atualizar pip e instalar dependências do modelo dentro do VENV
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Fix de compatibilidade (necessário para diffusers)
RUN pip install --no-cache-dir "numpy<2" "diffusers<0.31"

# Copiar requirements customizado (runpod, boto3, etc) e instalar no VENV
COPY requirements.txt /tmp/custom_reqs.txt
RUN pip install --no-cache-dir -r /tmp/custom_reqs.txt

# Instalar o pacote do modelo em modo editável (dentro do VENV)
RUN pip install --no-cache-dir -e .

# Compilar extensões CUDA (textura) dentro do VENV
RUN cd hy3dgen/texgen/custom_rasterizer && python setup.py install
RUN cd hy3dgen/texgen/differentiable_renderer && python setup.py install

# Limpeza de cache para reduzir tamanho do VENV
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 wget curl \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copiar o VENV completo (com todas as dependências e binários compilados)
COPY --from=builder /opt/venv /opt/venv

# Copiar código-fonte do Hunyuan3D-2.1
COPY --from=builder /Hunyuan3D-2 /Hunyuan3D-2

# Criar diretório do volume persistente
RUN mkdir -p /runpod-volume

# Copiar handler para dentro do diretório do modelo
COPY handler.py /Hunyuan3D-2/handler.py

# Definir diretório de trabalho
WORKDIR /Hunyuan3D-2

# Testar o handler para garantir que o modelo e dependências estão funcionando corretamente
COPY test_handler.py /Hunyuan3D-2/test_handler.py
RUN python /Hunyuan3D-2/test_handler.py

# Comando de inicialização
CMD ["python", "-u", "/Hunyuan3D-2/handler.py"]