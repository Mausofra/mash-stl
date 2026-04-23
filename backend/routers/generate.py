"""
Router principal: POST /generate
Aceita imagem e inicia a geração do mesh 3D via Hunyuan3D-2.
"""
import uuid
import asyncio
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from config import get_settings
from services.ollama import image_to_prompt
from services.preprocess import preprocess_image
from services.mesh import detect_mesh_format, extension_for_mesh
from services.runpod import generate_mesh
from routers.jobs import create_job, update_job

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}


@router.post("/generate")
async def generate(
    prompt: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Gera um modelo 3D a partir de uma imagem de referência.
    Opcionalmente aceita um prompt descritivo junto com a imagem.
    Se só a imagem for enviada, o Ollama gera o prompt automaticamente.
    """
    if not image:
        raise HTTPException(400, "Envie uma imagem de referência.")

    mime = image.content_type or ""
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, f"Formato de imagem não suportado: {mime}")

    settings = get_settings()
    image_bytes = await image.read()

    if len(image_bytes) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, f"Imagem excede o limite de {settings.max_upload_size_mb} MB.")

    # Pre-processamento leve para aumentar consistencia do resultado 3D.
    image_bytes = preprocess_image(
        image_bytes,
        max_dim=settings.preprocess_max_dim,
        autocrop=settings.preprocess_autocrop,
        pad_ratio=settings.preprocess_pad_ratio,
        apply_autocontrast=settings.preprocess_autocontrast,
    )

    job_id = str(uuid.uuid4())
    create_job(job_id, status="processing", progress=0, error=None)

    asyncio.create_task(_run_job(job_id, prompt, image_bytes, mime))

    return JSONResponse({"job_id": job_id, "status": "processing"})


async def _run_job(job_id: str, prompt: Optional[str], image_bytes: bytes, mime: str):
    try:
        if not prompt:
            update_job(job_id, progress=10)
            logger.info("[%s] Descrevendo imagem via Ollama...", job_id)
            prompt = await image_to_prompt(image_bytes, mime)

        update_job(job_id, progress=20)
        logger.info("[%s] Enviando para Hunyuan3D-2: %s", job_id, prompt[:80] if prompt else "")

        mesh_bytes = await generate_mesh(image_bytes)
        mesh_fmt = detect_mesh_format(mesh_bytes)
        mesh_ext = extension_for_mesh(mesh_fmt)

        update_job(
            job_id,
            status="completed",
            progress=100,
            mesh_bytes=mesh_bytes,
            filename=f"{job_id}.{mesh_ext}",
        )
        logger.info("[%s] Mesh gerado com sucesso (%d bytes).", job_id, len(mesh_bytes))

    except Exception as exc:
        logger.error("[%s] Erro: %s", job_id, exc)
        update_job(job_id, status="failed", error=str(exc))
