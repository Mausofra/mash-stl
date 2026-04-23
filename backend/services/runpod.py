"""
Serviço RunPod — integração com endpoint de geração de mesh 3D.

Fluxo:
  1. POST /runsync  -> { id, status, output }
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
    """Retorna headers de autorização para o endpoint Hunyuan3D-2."""
    settings = get_settings()
    key = settings.hunyuan3d_runpod_key or settings.runpod_api_key

    if not key:
        raise RuntimeError(
            "Chave do RunPod para Hunyuan3D-2 não configurada. "
            "Defina HUNYUAN3D_RUNPOD_KEY ou RUNPOD_API_KEY."
        )

    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _status_url(runpod_url: str, job_id: str) -> str:
    """Converte URL /runsync para URL /status/{id}."""
    if runpod_url.endswith("/runsync"):
        return runpod_url[: -len("/runsync")] + f"/status/{job_id}"
    return f"{runpod_url.rstrip('/')}/status/{job_id}"


def _extract_mesh(output: dict) -> bytes:
    """Extrai bytes do mesh a partir do output do RunPod."""
    if "error" in output:
        raise RuntimeError(f"Handler retornou erro: {output['error']}")

    mesh_b64 = output.get("mesh_b64")
    if not mesh_b64:
        raise RuntimeError(f"Output inesperado do handler: {output}")

    return base64.b64decode(mesh_b64)


async def _poll_job_status(poll_url: str, job_id: str, headers: dict) -> bytes:
    """Faz polling do status do job até completar, falhar ou timeout."""
    settings = get_settings()
    poll_interval = settings.runpod_poll_interval
    max_wait = settings.runpod_max_wait

    elapsed = 0

    async with httpx.AsyncClient(timeout=30) as client:
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            poll_resp = await client.get(poll_url, headers=headers)
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


async def generate_mesh_hunyuan3d(
    image_bytes: bytes,
    format: str = "glb",
    texture: bool = True,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.0,
    octree_resolution: int = 256,
) -> bytes:
    """
    Recebe bytes de imagem, chama o endpoint RunPod Hunyuan3D-2
    e retorna os bytes do arquivo 3D gerado (GLB ou OBJ).
    """
    settings = get_settings()

    if not settings.hunyuan3d_runpod_url:
        raise RuntimeError(
            "HUNYUAN3D_RUNPOD_URL não configurado no .env. "
            "Crie o endpoint Hunyuan3D-2 no RunPod e adicione a URL."
        )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = _headers()

    job_input = {
        "input": {
            "image": image_b64,
            "format": format.lower(),
            "texture": texture,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "octree_resolution": octree_resolution,
        }
    }

    if format.lower() not in ("glb", "obj"):
        raise ValueError("Formato inválido. Use 'glb' ou 'obj'.")

    if num_inference_steps < 1 or num_inference_steps > 200:
        raise ValueError("num_inference_steps deve estar entre 1 e 200.")

    if octree_resolution not in (128, 256, 512):
        raise ValueError("octree_resolution deve ser 128, 256 ou 512.")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            settings.hunyuan3d_runpod_url,
            headers=headers,
            json=job_input,
        )
        resp.raise_for_status()
        data = resp.json()

    job_id = data.get("id")
    status = data.get("status")
    output = data.get("output")

    if status == "COMPLETED" and output:
        return _extract_mesh(output)

    if not job_id:
        raise RuntimeError(f"RunPod não retornou job ID. Resposta: {data}")

    poll_url = _status_url(settings.hunyuan3d_runpod_url, job_id)
    return await _poll_job_status(poll_url, job_id, headers)


async def generate_mesh(image_bytes: bytes, **hunyuan3d_kwargs) -> bytes:
    """Função principal para geração de mesh via Hunyuan3D-2."""
    return await generate_mesh_hunyuan3d(image_bytes, **hunyuan3d_kwargs)


async def generate_mesh_compat(image_bytes: bytes) -> bytes:
    """Compatibilidade com código existente que espera generate_mesh(image_bytes)."""
    return await generate_mesh(image_bytes)
