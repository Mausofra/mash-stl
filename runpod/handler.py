"""
RunPod Serverless Handler — Hunyuan3D-2 (Alta Qualidade)

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
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Carrega pipelines uma única vez (warm start)
# ─────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.getenv("HUNYUAN3D_WEIGHTS", "tencent/Hunyuan3D-2")
SHAPE_PIPELINE = None
PAINT_PIPELINE = None


def _load_pipelines():
    """Carrega shape + texture pipelines na primeira chamada."""
    global SHAPE_PIPELINE, PAINT_PIPELINE

    if SHAPE_PIPELINE is not None:
        return

    logger.info("Carregando Hunyuan3D-DiT (shape)...")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    SHAPE_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        WEIGHTS_PATH,
        subfolder="hunyuan3d-dit-v2-0",
    )
    logger.info("Shape pipeline carregado.")

    logger.info("Carregando Hunyuan3D-Paint (textura)...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    PAINT_PIPELINE = Hunyuan3DPaintPipeline.from_pretrained(WEIGHTS_PATH)
    logger.info("Paint pipeline carregado.")


def _decode_image(image_b64: str):
    """Decodifica base64 para PIL Image."""
    from PIL import Image
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _export_mesh(mesh, output_format: str, output_dir: str) -> Path:
    """Exporta trimesh para o formato solicitado."""
    output_path = Path(output_dir) / f"output.{output_format}"
    mesh.export(str(output_path))
    return output_path


def handler(job):
    """
    Handler principal do RunPod Serverless.

    Input esperado:
        {
            "input": {
                "image": "<base64 da imagem PNG/JPG>",
                "format": "glb",          // opcional, "glb" ou "obj"
                "texture": true,          // opcional, default true
                "num_inference_steps": 50  // opcional, steps do DiT
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

    # ── Validação ──
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "Campo 'image' obrigatório (base64 de imagem PNG/JPG)."}

    output_format = job_input.get("format", "glb").lower()
    if output_format not in ("glb", "obj"):
        return {"error": "Formato inválido. Use 'glb' ou 'obj'."}

    with_texture = job_input.get("texture", True)
    num_steps = job_input.get("num_inference_steps", 50)

    output_dir = None

    try:
        # 0. Carrega pipelines (warm start após primeira chamada)
        _load_pipelines()

        # 1. Decodifica imagem
        image = _decode_image(image_b64)
        logger.info("Imagem decodificada: %s", image.size)

        # 2. Gera shape (mesh sem textura)
        logger.info("Gerando shape (steps=%d)...", num_steps)
        mesh = SHAPE_PIPELINE(
            image=image,
            num_inference_steps=num_steps,
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

        # 5. Codifica em base64
        with open(mesh_path, "rb") as f:
            mesh_b64 = base64.b64encode(f.read()).decode("utf-8")

        size_kb = mesh_path.stat().st_size / 1024
        logger.info("Mesh exportado: %s (%.1f KB)", mesh_path.name, size_kb)

        return {
            "mesh_b64": mesh_b64,
            "format": output_format,
            "filename": mesh_path.name,
        }

    except Exception as exc:
        logger.exception("Erro durante geração do mesh")
        return {"error": str(exc)}

    finally:
        if output_dir and os.path.isdir(output_dir):
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
#  Entrypoint RunPod
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Inicializando handler Hunyuan3D-2...")
    _load_pipelines()
    logger.info("Pipelines prontos. Aguardando jobs...")
    runpod.serverless.start({"handler": handler})
