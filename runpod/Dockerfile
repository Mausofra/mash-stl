# ─────────────────────────────────────────────────────────────
#  Hunyuan3D-2 shape generation — RunPod serverless
#  Fase 1: shape only (sem textura), pesos via HuggingFace Hub
# ─────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── Deps de sistema ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Shallow clone do Hunyuan3D-2 ──
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /Hunyuan3D-2
WORKDIR /Hunyuan3D-2

# ── Apenas deps necessárias (torch já vem na base) ──
RUN pip install --no-cache-dir \
    diffusers einops opencv-python-headless transformers \
    omegaconf tqdm trimesh pymeshlab pygltflib xatlas \
    accelerate pybind11 huggingface_hub Pillow

# ── Instala hy3dgen (--no-deps evita reinstalar torch/gradio/rembg) ──
RUN pip install --no-cache-dir --no-deps -e .

# ── RunPod SDK ──
RUN pip install --no-cache-dir runpod

# ── Handler ──
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
