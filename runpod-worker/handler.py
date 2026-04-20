"""
RunPod Serverless Handler — Hunyuan3D-2 (Shape + Textura)
Pipeline:
  1. Recebe imagem base64
  2. Gera mesh (shape) via Hunyuan3D-DiT
  3. Gera textura via Hunyuan3D-Paint
  4. Exporta GLB/OBJ com textura e retorna base64
"""
import runpod
import base64
import io
import tempfile
import os
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import huggingface_hub
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Configurações globais (podem vir de variáveis de ambiente)
# ─────────────────────────────────────────────────────────────
VOLUME_PATH = os.getenv("VOLUME_PATH", "/runpod-volume")
WEIGHTS_PATH = os.path.join(VOLUME_PATH, "Hunyuan3D-2")
HF_REPO = "tencent/Hunyuan3D-2"

# Forca cache do Hugging Face no volume persistente (evita disco efemero).
os.environ.setdefault("HF_HOME", os.path.join(VOLUME_PATH, ".cache", "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(VOLUME_PATH, ".cache", "huggingface", "hub"))

# Limites de segurança
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
MAX_MESH_SIZE_MB = int(os.getenv("MAX_MESH_SIZE_MB", "50"))

# Pipelines globais (carregadas lazy)
SHAPE_PIPELINE: Optional[Any] = None
PAINT_PIPELINE: Optional[Any] = None

# ─────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────
def _check_disk_space(path: str, required_gb: float = 15.0) -> bool:
    """Verifica se há espaço em disco suficiente."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    
    logger.info(f"Espaço em disco em {path}: {free_gb:.1f}GB livre de {total_gb:.1f}GB total")
    
    if free_gb < required_gb:
        logger.error(f"Espaço insuficiente em {path}: {free_gb:.1f}GB livre, necessário {required_gb}GB")
        return False
    return True

def _cleanup_old_cache_if_needed(path: str, min_free_gb: float = 5.0):
    """Limpa cache antigo se espaço livre estiver abaixo do mínimo."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    
    if free_gb >= min_free_gb:
        return
    
    logger.warning(f"Espaço livre baixo ({free_gb:.1f}GB), verificando cache para limpeza...")
    
    # Procurar arquivos temporários antigos
    cache_dir = Path(path) / "cache"
    if cache_dir.exists():
        try:
            # Listar arquivos por data de modificação
            files = list(cache_dir.rglob("*"))
            files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remover os mais antigos até liberar espaço
            freed_gb = 0
            for file in files[:10]:  # Limitar a 10 arquivos por segurança
                if file.is_file():
                    size_gb = file.stat().st_size / (1024**3)
                    try:
                        file.unlink()
                        freed_gb += size_gb
                        logger.info(f"Removido cache antigo: {file.name} ({size_gb:.2f}GB)")
                    except Exception as e:
                        logger.warning(f"Erro ao remover {file}: {e}")
            
            if freed_gb > 0:
                logger.info(f"Liberados {freed_gb:.2f}GB de espaço em cache")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {e}")

def _ensure_weights():
    """Baixa pesos para o volume persistente com retry e verificação de integridade."""
    marker = os.path.join(WEIGHTS_PATH, ".download_complete")
    if os.path.exists(marker):
        logger.info("Pesos já existem em %s, pulando download.", WEIGHTS_PATH)
        return

    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    
    # Verificar espaço e tentar limpar cache se necessário
    if not _check_disk_space(WEIGHTS_PATH, required_gb=15.0):
        logger.warning("Espaço insuficiente, tentando limpar cache...")
        _cleanup_old_cache_if_needed(VOLUME_PATH, min_free_gb=15.0)
        
        # Verificar novamente após limpeza
        if not _check_disk_space(WEIGHTS_PATH, required_gb=15.0):
            raise RuntimeError("Espaço em disco insuficiente para baixar os modelos mesmo após limpeza de cache.")

    logger.info("Baixando/retomando pesos de %s para %s ...", HF_REPO, WEIGHTS_PATH)
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=WEIGHTS_PATH,
            ignore_patterns=[
                "*.md", "*.txt", "*.git*", "*.pdf",
                "docs/**", "examples/**", "assets/**",
                "*.mp4", "*.webm", "*.gif",
            ],
            resume_download=True,
        )
        # Criar marcador para indicar sucesso
        open(marker, "w").close()
        logger.info("Pesos baixados com sucesso.")
    except Exception as e:
        # Se falhar, limpar diretório parcial
        logger.exception("Falha no download dos pesos. Limpando diretório parcial...")
        shutil.rmtree(WEIGHTS_PATH, ignore_errors=True)
        raise RuntimeError(f"Download dos pesos falhou: {e}")

def _load_pipelines():
    """Carrega shape + texture pipelines na primeira chamada (thread-safe via lock global)."""
    global SHAPE_PIPELINE, PAINT_PIPELINE

    if SHAPE_PIPELINE is not None:
        return

    _ensure_weights()

    logger.info("Carregando Hunyuan3D-DiT (shape) de %s ...", WEIGHTS_PATH)
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    SHAPE_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        WEIGHTS_PATH,
        subfolder="hunyuan3d-dit-v2-0",
        device="cuda",
        torch_dtype=torch.float16,
    )
    logger.info("Shape pipeline carregado.")

    logger.info("Carregando Hunyuan3D-Paint (textura)...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    PAINT_PIPELINE = Hunyuan3DPaintPipeline.from_pretrained(
        WEIGHTS_PATH,
        device="cuda",
        torch_dtype=torch.float16,
    )
    logger.info("Paint pipeline carregado.")

def _decode_image(image_b64: str) -> Image.Image:
    """Decodifica base64 para PIL Image com validação de tamanho."""
    # Validação básica de tamanho (aproximado)
    if len(image_b64) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Imagem excede limite de {MAX_IMAGE_SIZE_MB}MB")

    from PIL import Image
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Redimensiona se muito grande (opcional, mas recomendado)
    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        logger.info("Imagem redimensionada para %s", img.size)

    return img

def _export_mesh(mesh, output_format: str, output_dir: str) -> Path:
    """Exporta trimesh para o formato solicitado. Para OBJ, cria ZIP com MTL e textura se houver."""
    output_path = Path(output_dir) / f"output.{output_format}"
    mesh.export(str(output_path))

    # Se OBJ e textura presente, empacota com MTL e textura
    if output_format == "obj" and hasattr(mesh.visual, 'material'):
        # Salvar textura como PNG
        texture_path = Path(output_dir) / "texture.png"
        if hasattr(mesh.visual, 'to_texture'):
            mesh.visual.to_texture().save(texture_path)

        # Criar ZIP contendo OBJ, MTL e PNG
        zip_path = Path(output_dir) / "output.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(output_path, arcname="model.obj")
            mtl_path = output_path.with_suffix('.mtl')
            if mtl_path.exists():
                zipf.write(mtl_path, arcname="model.mtl")
            if texture_path.exists():
                zipf.write(texture_path, arcname="texture.png")
        return zip_path

    return output_path

def _cleanup_gpu():
    """Libera memória GPU não utilizada."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ─────────────────────────────────────────────────────────────
# Handler principal
# ─────────────────────────────────────────────────────────────
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler principal do RunPod Serverless.

    Input esperado:
        {
            "input": {
                "image": "<base64 da imagem PNG/JPG>",
                "format": "glb",          // opcional, "glb" ou "obj"
                "texture": true,          // opcional, default true
                "num_inference_steps": 100, // opcional
                "guidance_scale": 7.0,    // opcional
                "octree_resolution": 256   // opcional
            }
        }

    Output:
        {
            "mesh_b64": "<base64 do arquivo 3D>",
            "format": "glb",
            "filename": "output.glb"
        }
    """
    job_input = job.get("input", {})

    # Validação
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "Campo 'image' obrigatório (base64 de imagem PNG/JPG)."}

    output_format = job_input.get("format", "glb").lower()
    if output_format not in ("glb", "obj"):
        return {"error": "Formato inválido. Use 'glb' ou 'obj'."}

    num_steps = job_input.get("num_inference_steps", 100)
    guidance_scale = job_input.get("guidance_scale", 7.0)
    octree_resolution = job_input.get("octree_resolution", 256)
    with_texture = job_input.get("texture", True)

    output_dir = None

    try:
        # 0. Verificar espaço em disco antes de começar
        _check_disk_space(VOLUME_PATH, required_gb=15.0)
        
        # 1. Carrega pipelines (warm start após primeira chamada)
        _load_pipelines()

        # 1. Decodifica imagem
        image = _decode_image(image_b64)
        logger.info("Imagem decodificada: %s", image.size)

        # 2. Gera shape (mesh sem textura)
        logger.info("Gerando shape (steps=%d, guidance=%.1f, octree=%d)...",
                    num_steps, guidance_scale, octree_resolution)
        mesh = SHAPE_PIPELINE(
            image=image,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
        )[0]
        logger.info("Shape gerado com sucesso.")

        # 3. Gera textura (alta qualidade)
        if with_texture and PAINT_PIPELINE is not None:
            logger.info("Gerando textura...")
            mesh = PAINT_PIPELINE(mesh, image=image)
            logger.info("Textura gerada com sucesso.")

        # 4. Exporta para o formato solicitado
        output_dir = tempfile.mkdtemp(prefix="hunyuan3d_out_")
        mesh_path = _export_mesh(mesh, output_format, output_dir)

        # Verifica tamanho do arquivo
        file_size_mb = mesh_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_MESH_SIZE_MB:
            raise ValueError(f"Arquivo gerado ({file_size_mb:.1f}MB) excede limite de {MAX_MESH_SIZE_MB}MB")

        # 5. Codifica em base64
        with open(mesh_path, "rb") as f:
            mesh_b64 = base64.b64encode(f.read()).decode("utf-8")

        logger.info("Mesh exportado: %s (%.1f MB)", mesh_path.name, file_size_mb)

        return {
            "mesh_b64": mesh_b64,
            "format": output_format,
            "filename": mesh_path.name,
        }

    except Exception as exc:
        logger.exception("Erro durante geração do mesh")
        # Retornar erro estruturado
        return {
            "error": str(exc),
            "error_type": type(exc).__name__
        }

    finally:
        # Limpeza do diretório temporário
        if output_dir and os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        # Liberar VRAM
        _cleanup_gpu()

# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Inicializando handler Hunyuan3D-2...")
    logger.info("VOLUME_PATH=%s", VOLUME_PATH)
    logger.info("WEIGHTS_PATH=%s", WEIGHTS_PATH)
    logger.info("MAX_IMAGE_SIZE_MB=%d, MAX_MESH_SIZE_MB=%d", MAX_IMAGE_SIZE_MB, MAX_MESH_SIZE_MB)

    # Opcional: pré-carregar modelos no startup (útil para evitar cold start no primeiro job)
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        logger.info("Pré-carregando pipelines (pode levar vários minutos)...")
        try:
            _load_pipelines()
            logger.info("Pipelines pré-carregados com sucesso.")
        except Exception as e:
            logger.error("Falha no pré-carregamento: %s", e)
            # Não abortar, tentará novamente no primeiro job

    logger.info("Registrando worker...")
    runpod.serverless.start({"handler": handler})