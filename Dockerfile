# ─────────────────────────────────────────────────────────────
#  Hunyuan3D-2 — RunPod serverless (shape + textura)
# ─────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA=1

# ── Deps de sistema ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1 libglib2.0-0 libgomp1 ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ── Clone Hunyuan3D-2 ──
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /Hunyuan3D-2
WORKDIR /Hunyuan3D-2

# ── Deps Python (torch já vem na base) ──
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# ── Compila custom rasterizer + differentiable renderer (textura) ──
RUN cd hy3dgen/texgen/custom_rasterizer && \
    python setup.py install && \
    cd /Hunyuan3D-2

RUN cd hy3dgen/texgen/differentiable_renderer && \
    python setup.py install && \
    cd /Hunyuan3D-2

# ── RunPod SDK ──
COPY runpod/requirements.txt /tmp/runpod_requirements.txt
RUN pip install --no-cache-dir -r /tmp/runpod_requirements.txt

# ── Pré-download dos pesos (shape + texture + delight) ──
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tencent/Hunyuan3D-2', local_dir='/Hunyuan3D-2/weights', ignore_patterns=['*.md', '*.txt', '*.git*']); print('Pesos baixados.')"

# ── Handler ──
COPY runpod/handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
