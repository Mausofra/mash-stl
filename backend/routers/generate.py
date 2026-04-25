"""
Router principal: POST /generate
Aceita imagem e inicia a geração do mesh 3D via Hunyuan3D-2.
"""
import uuid
import asyncio
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List

from config import get_settings
from services.ollama import image_to_prompt
from services.preprocess import preprocess_image
from services.mesh import detect_mesh_format, extension_for_mesh
from services.runpod import generate_mesh
from routers.jobs import create_job, update_job

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}

QUALITY_PRESETS = {
    "rapido": dict(texture=False, num_inference_steps=30, octree_resolution=128),
    "padrao": dict(texture=True,  num_inference_steps=50, octree_resolution=256),
    "alta":   dict(texture=True,  num_inference_steps=100, octree_resolution=256),
}

QUALITY_EXPECTED_S = {"rapido": 60, "padrao": 180, "alta": 300}


@router.post("/generate")
async def generate(
    prompt: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    extra_images: Optional[List[UploadFile]] = File(None),
    quality: str = Form("padrao"),
):
    """
    Gera um modelo 3D a partir de uma ou mais imagens de referência.
    extra_images: até 3 ângulos adicionais para multi-view.
    quality: 'rapido' | 'padrao' | 'alta'
    """
    if not image:
        raise HTTPException(400, "Envie uma imagem de referência.")

    mime = image.content_type or ""
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, f"Formato de imagem não suportado: {mime}")

    if quality not in QUALITY_PRESETS:
        raise HTTPException(400, f"quality inválido. Use: {', '.join(QUALITY_PRESETS)}")

    settings = get_settings()
    size_limit = settings.max_upload_size_mb * 1024 * 1024

    image_bytes = await image.read()
    if len(image_bytes) > size_limit:
        raise HTTPException(413, f"Imagem excede o limite de {settings.max_upload_size_mb} MB.")

    # Coleta e valida imagens extras (multi-view)
    extra_bytes_list: List[bytes] = []
    for extra in (extra_images or [])[:3]:
        if extra.content_type not in ALLOWED_MIME:
            continue
        b = await extra.read()
        if len(b) <= size_limit:
            extra_bytes_list.append(b)

    def _preprocess(raw: bytes) -> bytes:
        return preprocess_image(
            raw,
            max_dim=settings.preprocess_max_dim,
            autocrop=settings.preprocess_autocrop,
            pad_ratio=settings.preprocess_pad_ratio,
            apply_autocontrast=settings.preprocess_autocontrast,
            use_rembg=settings.preprocess_rembg,
        )

    image_bytes = _preprocess(image_bytes)
    extra_processed = [_preprocess(b) for b in extra_bytes_list]

    job_id = str(uuid.uuid4())
    create_job(job_id, status="processing", progress=0, error=None)

    asyncio.create_task(
        _run_job(job_id, prompt, image_bytes, extra_processed, mime, quality)
    )

    return JSONResponse({"job_id": job_id, "status": "processing"})


async def _run_job(job_id: str, prompt: Optional[str], image_bytes: bytes, extra_images: List[bytes], mime: str, quality: str):
    try:
        mesh_params = QUALITY_PRESETS[quality]
        expected_s = QUALITY_EXPECTED_S[quality]

        if not prompt:
            update_job(job_id, progress=10)
            logger.info("[%s] Descrevendo imagem via Ollama...", job_id)
            prompt = await image_to_prompt(image_bytes, mime)

        update_job(job_id, progress=20)
        logger.info("[%s] Enviando para Hunyuan3D-2 (quality=%s)...", job_id, quality)

        def _on_progress(pct: int):
            update_job(job_id, progress=pct)

        mesh_bytes = await generate_mesh(
            image_bytes,
            extra_images=extra_images,
            prompt=prompt,
            on_progress=_on_progress,
            expected_seconds=expected_s,
            **mesh_params,
        )

        update_job(job_id, progress=95)
        mesh_fmt = detect_mesh_format(mesh_bytes)
        mesh_ext = extension_for_mesh(mesh_fmt)
        filename = f"{job_id}.{mesh_ext}"

        update_job(
            job_id,
            status="completed",
            progress=100,
            mesh_bytes=mesh_bytes,
            filename=filename,
        )
        logger.info("[%s] Mesh gerado com sucesso (%d bytes).", job_id, len(mesh_bytes))

    except Exception as exc:
        logger.error("[%s] Erro: %s", job_id, exc)
        update_job(job_id, status="failed", error=str(exc))
