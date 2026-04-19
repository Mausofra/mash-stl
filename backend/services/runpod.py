"""
Serviço RunPod — envia imagem base64 para o endpoint
Hunyuan3D-2 serverless e retorna o mesh em base64.

Fluxo:
  1. POST /runsync  → { id, status, output }   (síncrono até 90s)
  2. Se status == "IN_PROGRESS", faz poll em GET /status/{id}
  3. Quando status == "COMPLETED" retorna output.mesh_b64
"""
import asyncio
import base64
import httpx
import logging
from config import get_settings

logger = logging.getLogger(__name__)


def _headers() -> dict:
    settings = get_settings()
    key = settings.instantmesh_runpod_key or settings.runpod_api_key
    if not key:
        raise RuntimeError(
            "Chave do RunPod não configurada. Defina INSTANTMESH_RUNPOD_KEY ou RUNPOD_API_KEY."
        )
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _status_url(runpod_url: str, job_id: str) -> str:
    if runpod_url.endswith("/runsync"):
        return runpod_url[: -len("/runsync")] + f"/status/{job_id}"
    return f"{runpod_url.rstrip('/')}/status/{job_id}"


async def generate_mesh(image_bytes: bytes) -> bytes:
    """
    Recebe bytes de imagem, chama o endpoint RunPod InstantMesh
    e retorna os bytes do arquivo .obj gerado.
    """
    settings = get_settings()

    if not settings.instantmesh_runpod_url:
        raise RuntimeError(
            "INSTANTMESH_RUNPOD_URL não configurado no .env. "
            "Crie o endpoint Hunyuan3D-2 no RunPod e adicione a URL."
        )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    async with httpx.AsyncClient(timeout=120) as client:
        # 1. Dispara o job (runsync espera até 90s antes de retornar IN_PROGRESS)
        resp = await client.post(
            settings.instantmesh_runpod_url,
            headers=_headers(),
            json={"input": {"image": image_b64}},
        )
        resp.raise_for_status()
        data = resp.json()

    job_id = data.get("id")
    status = data.get("status")
    output = data.get("output")

    # 2. Se já terminou (runsync rápido)
    if status == "COMPLETED" and output:
        return _extract_mesh(output)

    # 3. Se ainda está rodando, faz poll
    if not job_id:
        raise RuntimeError(f"RunPod não retornou job ID. Resposta: {data}")

    poll_url = _status_url(settings.instantmesh_runpod_url, job_id)
    elapsed = 0

    settings = get_settings()
    poll_interval = settings.runpod_poll_interval
    max_wait = settings.runpod_max_wait

    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            poll_resp = await client.get(poll_url, headers=_headers())
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

            status = poll_data.get("status")
            logger.info("Job %s status: %s (%ds)", job_id, status, elapsed)

            if status == "COMPLETED":
                return _extract_mesh(poll_data.get("output", {}))

            if status in ("FAILED", "CANCELLED"):
                error = poll_data.get("error", "sem detalhes")
                raise RuntimeError(f"Job RunPod {status}: {error}")

    raise TimeoutError(f"Job {job_id} não concluiu em {max_wait}s.")


def _extract_mesh(output: dict) -> bytes:
    """Extrai bytes do mesh a partir do output do RunPod."""
    if "error" in output:
        raise RuntimeError(f"Handler retornou erro: {output['error']}")

    mesh_b64 = output.get("mesh_b64")
    if not mesh_b64:
        raise RuntimeError(f"Output inesperado do handler: {output}")

    return base64.b64decode(mesh_b64)
