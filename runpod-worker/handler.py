"""
RunPod Serverless Handler — Hunyuan3D-2.1 (Shape + Textura PBR)
Pipeline:
  1. Recebe imagem base64
  2. Gera mesh (shape) via Hunyuan3D-DiT v2.1
  3. Gera textura PBR via Hunyuan3D-Paint v2.1
  4. Exporta GLB/OBJ e faz upload para Cloudflare R2 (S3)
  5. Retorna URL pré-assinada (Pre-signed URL) válida por 24h
"""
import sys
import os

# ─────────────────────────────────────────────────────────────
# Fix de Arquitetura Hunyuan3D v2.1
# Injeta as pastas da nova arquitetura no PATH do Python
# para que os imports funcionem localmente sem o setup.py
# ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.getcwd(), 'hy3dshape'))
sys.path.insert(0, os.path.join(os.getcwd(), 'hy3dpaint'))

import gc
import runpod
import base64
import io
import tempfile
import logging
import shutil
import zipfile
import threading
import uuid
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import boto3
from botocore.exceptions import ClientError
from PIL import Image
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Configurações globais de Infraestrutura
# ─────────────────────────────────────────────────────────────
VOLUME_PATH = os.getenv("VOLUME_PATH", "/runpod-volume")
WEIGHTS_PATH = os.path.join(VOLUME_PATH, "Hunyuan3D-2.1")
HF_REPO = "tencent/Hunyuan3D-2.1"
HF_TOKEN = os.getenv("HF_TOKEN")

import tempfile
_vol_tmp = os.path.join(VOLUME_PATH, "tmp")
try:
    os.makedirs(_vol_tmp, exist_ok=True)
    os.environ["TMPDIR"] = _vol_tmp
    os.environ["TEMP"] = _vol_tmp
    os.environ["TMP"] = _vol_tmp
    tempfile.tempdir = _vol_tmp
except OSError:
    pass

os.environ.setdefault("HF_HOME", os.path.join(VOLUME_PATH, ".cache", "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(VOLUME_PATH, ".cache", "huggingface", "hub"))

MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
MAX_MESH_SIZE_MB  = int(os.getenv("MAX_MESH_SIZE_MB", "100"))

# ─────────────────────────────────────────────────────────────
# Cloudflare R2
# ─────────────────────────────────────────────────────────────
R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL")
R2_BUCKET_NAME       = os.getenv("R2_BUCKET_NAME", "hunyuan-outputs")
AWS_ACCESS_KEY_ID    = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = None
if all([R2_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
    s3_client = boto3.client(
        's3',
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="auto"
    )
else:
    logger.warning("Credenciais do Cloudflare R2 incompletas. O upload falhará na inferência.")

# Pipelines globais e Locks
SHAPE_PIPELINE: Optional[Any] = None
PAINT_PIPELINE: Optional[Any] = None
_pipeline_lock  = threading.Lock()
_inference_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# Funções auxiliares de Infra
# ─────────────────────────────────────────────────────────────
def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _check_disk_space(path: str, required_gb: float = 20.0) -> bool:
    os.makedirs(path, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    logger.info(f"Espaço em disco em {path}: {free_gb:.1f}GB livre")
    if free_gb < required_gb:
        logger.error(f"Espaço insuficiente: {free_gb:.1f}GB livre, necessário {required_gb}GB")
        return False
    return True


def _cleanup_old_cache_if_needed(path: str, min_free_gb: float = 5.0):
    usage = shutil.disk_usage(path)
    if (usage.free / (1024**3)) >= min_free_gb:
        return
    cache_dir = Path(path) / "cache"
    if cache_dir.exists():
        try:
            files = sorted(list(cache_dir.rglob("*")), key=lambda x: x.stat().st_mtime)
            for file in files[:10]:
                if file.is_file():
                    file.unlink()
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")


def _ensure_weights():
    marker = os.path.join(WEIGHTS_PATH, ".download_complete")
    if os.path.exists(marker):
        return

    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    if not _check_disk_space(WEIGHTS_PATH, required_gb=20.0):
        _cleanup_old_cache_if_needed(VOLUME_PATH, min_free_gb=20.0)
        if not _check_disk_space(WEIGHTS_PATH, required_gb=20.0):
            raise RuntimeError("Espaço em disco insuficiente para baixar os modelos.")

    logger.info("Baixando pesos de %s para %s ...", HF_REPO, WEIGHTS_PATH)
    try:
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=WEIGHTS_PATH,
            ignore_patterns=["*.md", "*.txt", "*.git*", "*.pdf", "docs/**", "examples/**", "assets/**", "*.mp4", "*.webm", "*.gif"],
            resume_download=True,
            token=HF_TOKEN,
        )
        Path(marker).touch()
    except Exception as e:
        shutil.rmtree(WEIGHTS_PATH, ignore_errors=True)
        raise RuntimeError(f"Download dos pesos falhou: {e}")


# ─────────────────────────────────────────────────────────────
# Carregamento e descarregamento dos pipelines
# Estratégia de VRAM: SHAPE (~20 GB) e PAINT (~6 GB) não cabem
# juntos em uma RTX 4090 (24 GB). Carregamos um por vez:
#   1. Carregar SHAPE → rodar → descarregar completamente
#   2. Carregar PAINT → rodar → manter em cache para próximo job
# ─────────────────────────────────────────────────────────────
def _load_shape():
    global SHAPE_PIPELINE
    with _pipeline_lock:
        if SHAPE_PIPELINE is not None:
            return
        _ensure_weights()
        logger.info("Carregando SHAPE_PIPELINE...")
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        SHAPE_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            WEIGHTS_PATH, subfolder="hunyuan3d-dit-v2-1", device="cuda", torch_dtype=torch.float16
        )
        logger.info("SHAPE_PIPELINE pronto.")


def _unload_shape():
    global SHAPE_PIPELINE
    with _pipeline_lock:
        if SHAPE_PIPELINE is None:
            return
        logger.info("Descarregando SHAPE_PIPELINE da VRAM...")
        SHAPE_PIPELINE = None
        gc.collect()
        _cleanup_gpu()
        if torch.cuda.is_available():
            free_gb = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"VRAM reservada após unload: {free_gb:.2f} GB")


def _load_paint():
    global PAINT_PIPELINE
    with _pipeline_lock:
        if PAINT_PIPELINE is not None:
            return
        _ensure_weights()

        esrgan_path = os.path.join(os.getcwd(), 'ckpt', 'RealESRGAN_x4plus.pth')
        if not os.path.exists(esrgan_path):
            os.makedirs(os.path.dirname(esrgan_path), exist_ok=True)
            logger.info("Baixando RealESRGAN para o pipeline de textura PBR...")
            urllib.request.urlretrieve(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                esrgan_path,
            )

        logger.info("Carregando PAINT_PIPELINE...")
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        PAINT_PIPELINE = Hunyuan3DPaintPipeline(
            Hunyuan3DPaintConfig(max_num_view=6, resolution=1024)
        )
        logger.info("PAINT_PIPELINE pronto.")


# ─────────────────────────────────────────────────────────────
# Decodificação / Exportação
# ─────────────────────────────────────────────────────────────
def _decode_image(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64)
        if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Imagem excede limite de {MAX_IMAGE_SIZE_MB}MB")
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        if max(img.size) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        raise ValueError(f"Falha na decodificação da imagem: {e}")


def _export_mesh(mesh, output_format: str, output_dir: str) -> Path:
    output_path = Path(output_dir) / f"output.{output_format}"
    mesh.export(str(output_path))

    if output_format == "obj":
        zip_path = Path(output_dir) / "output.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(output_path, arcname="model.obj")
            mtl_path = output_path.with_suffix('.mtl')
            if mtl_path.exists():
                zipf.write(mtl_path, arcname="model.mtl")
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                material = mesh.visual.material
                if hasattr(material, 'image') and material.image is not None:
                    texture_path = Path(output_dir) / "texture.png"
                    material.image.save(str(texture_path))
                    zipf.write(texture_path, arcname="texture.png")
        return zip_path
    return output_path


# ─────────────────────────────────────────────────────────────
# Cloudflare R2
# ─────────────────────────────────────────────────────────────
def _upload_to_r2_and_get_url(file_path: Path, expiration_seconds: int = 86400) -> str:
    if s3_client is None:
        raise RuntimeError("Cliente S3 (Cloudflare R2) não foi inicializado. Verifique as credenciais.")

    object_name = f"outputs/{uuid.uuid4()}_{file_path.name}"

    try:
        logger.info(f"Enviando {file_path.name} para R2: {R2_BUCKET_NAME}/{object_name}")
        s3_client.upload_file(str(file_path), R2_BUCKET_NAME, object_name)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': R2_BUCKET_NAME, 'Key': object_name},
            ExpiresIn=expiration_seconds,
        )
        logger.info("Upload concluído.")
        return presigned_url
    except ClientError as e:
        logger.error(f"Erro R2: {e}")
        raise RuntimeError(f"Falha no upload para o R2: {e}")


# ─────────────────────────────────────────────────────────────
# Handler principal
# ─────────────────────────────────────────────────────────────
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})

    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "Campo 'image' obrigatório."}

    output_format = job_input.get("format", "glb").lower()
    if output_format not in ("glb", "obj"):
        return {"error": "Formato inválido. Use 'glb' ou 'obj'."}

    num_steps        = job_input.get("num_inference_steps", 100)
    guidance_scale   = job_input.get("guidance_scale", 7.0)
    octree_resolution = job_input.get("octree_resolution", 256)
    with_texture     = job_input.get("texture", True)

    output_dir = None

    try:
        if not _check_disk_space(VOLUME_PATH, required_gb=20.0):
            return {"error": "Espaço em disco insuficiente."}

        image = _decode_image(image_b64)

        paint_output_path = None  # preenchido se PAINT_PIPELINE retornar caminho

        with _inference_lock:
            # ── Shape ──────────────────────────────────────────
            _load_shape()
            logger.info("Iniciando inferência 3D (shape)...")
            mesh = SHAPE_PIPELINE(
                image=image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
            )[0]
            logger.info("Shape concluído.")

            # ── Textura ────────────────────────────────────────
            if with_texture:
                _unload_shape()

                _load_paint()
                logger.info("Aplicando textura PBR...")

                # Usa o diretório nativo do Hunyuan3D como cwd para os
                # intermediários do paint — o pipeline resolve textured_mesh.obj
                # relativo a dirname(mesh_path), que precisa ser local (não rede).
                PAINT_WORK_DIR = "/Hunyuan3D-2.1"
                os.makedirs(PAINT_WORK_DIR, exist_ok=True)
                temp_img_path  = os.path.join(PAINT_WORK_DIR, "_input_ref.png")
                temp_mesh_path = os.path.join(PAINT_WORK_DIR, "_shape_raw.obj")
                try:
                    image.save(temp_img_path)
                    mesh.export(temp_mesh_path)

                    paint_result = PAINT_PIPELINE(temp_mesh_path, image_path=temp_img_path)
                    logger.info("Textura PBR concluída.")

                    if isinstance(paint_result, str):
                        paint_output_path = paint_result
                        logger.info(f"PAINT retornou arquivo: {paint_output_path}")
                    else:
                        mesh = paint_result
                finally:
                    for _f in [temp_img_path, temp_mesh_path]:
                        try:
                            os.unlink(_f)
                        except OSError:
                            pass

        output_dir = tempfile.mkdtemp(prefix="hunyuan3d_out_")

        if paint_output_path is not None:
            # GLB já gerado pelo PAINT_PIPELINE — copiar para output_dir
            src = Path(paint_output_path)
            ext = src.suffix.lstrip('.') or "glb"
            mesh_path = Path(output_dir) / f"output.{ext}"
            shutil.copy2(str(src), str(mesh_path))
            returned_format = ext
        else:
            mesh_path = _export_mesh(mesh, output_format, output_dir)
            returned_format = "zip" if output_format == "obj" else output_format

        file_size_mb = mesh_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_MESH_SIZE_MB:
            raise ValueError(f"Arquivo gerado ({file_size_mb:.1f}MB) muito grande.")

        mesh_url = _upload_to_r2_and_get_url(mesh_path)

        return {
            "mesh_url": mesh_url,
            "format": returned_format,
            "filename": mesh_path.name,
        }

    except Exception as exc:
        logger.exception("Erro durante geração")
        return {"error": str(exc), "error_type": type(exc).__name__}

    finally:
        if output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        _cleanup_gpu()


if __name__ == "__main__":
    logger.info("Inicializando worker Serverless...")
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        logger.info("PRELOAD_MODELS=true — pré-carregando SHAPE_PIPELINE...")
        _load_shape()
    runpod.serverless.start({"handler": handler, "refresh_worker": False})
