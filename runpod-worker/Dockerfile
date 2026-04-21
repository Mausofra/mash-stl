# Multi-stage Dockerfile para Hunyuan3D-2 RunPod Worker
# Stage 1: Build CUDA extensions
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA=1

# Dependências de sistema para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 libgomp1 ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clonar repositório oficial
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /Hunyuan3D-2

WORKDIR /Hunyuan3D-2

# Instalar dependências Python do modelo
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Fix de compatibilidade
RUN pip install --no-cache-dir "numpy<2" "diffusers<0.31"

# Compilar extensões CUDA (textura)
RUN cd hy3dgen/texgen/custom_rasterizer && python setup.py install
RUN cd hy3dgen/texgen/differentiable_renderer && python setup.py install

# ─────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VOLUME_PATH=/runpod-volume \
    MAX_IMAGE_SIZE_MB=5 \
    MAX_MESH_SIZE_MB=50 \
    PRELOAD_MODELS=false

# Dependências de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas as extensões CUDA compiladas do builder
# (evita conflito entre ambientes Conda do builder e do runtime)
COPY --from=builder /opt/conda/lib/python3.10/site-packages/custom_rasterizer* \
    /opt/conda/lib/python3.10/site-packages/
COPY --from=builder /opt/conda/lib/python3.10/site-packages/differentiable_renderer* \
    /opt/conda/lib/python3.10/site-packages/

# Copiar código-fonte do Hunyuan3D-2 (sem os pacotes Python)
COPY --from=builder /Hunyuan3D-2 /Hunyuan3D-2

# Reinstalar dependências no ambiente runtime (correto e limpo)
RUN pip install --no-cache-dir -r /Hunyuan3D-2/requirements.txt && \
    pip install --no-cache-dir "numpy<2" "diffusers<0.31" && \
    pip install --no-cache-dir -e /Hunyuan3D-2

# Instalar SDK do RunPod
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Criar diretório do volume persistente
RUN mkdir -p /runpod-volume

# Copiar handler
COPY handler.py /handler.py

# Workdir aponta para o repositório (necessário para imports relativos do hy3dgen)
WORKDIR /Hunyuan3D-2

# Limpeza final
RUN apt-get clean && \
    pip cache purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD ["python", "-u", "/handler.py"]