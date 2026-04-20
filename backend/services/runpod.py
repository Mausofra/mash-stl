"""
Serviço RunPod — integração com endpoints de geração de mesh 3D.

Suporta:
1. InstantMesh (legado)
2. Hunyuan3D-2 (novo worker principal com shape + textura)

Fluxo:
  1. POST /runsync  → { id, status, output }   (síncrono até 90s)
  2. Se status == "IN_PROGRESS", faz poll em GET /status/{id}
  3. Quando status == "COMPLETED" retorna output.mesh_b64
"""
import asyncio
import base64
import httpx
import logging
from typing import Optional, Dict, Any
from config import get_settings

logger = logging.getLogger(__name__)


def _headers_for_endpoint(endpoint_type: str = "instantmesh") -> dict:
    """Retorna headers de autorização para o endpoint especificado."""
    settings = get_settings()
    
    if endpoint_type == "instantmesh":
        key = settings.instantmesh_runpod_key or settings.runpod_api_key
        if not key:
            raise RuntimeError(
                "Chave do RunPod para InstantMesh não configurada. "
                "Defina INSTANTMESH_RUNPOD_KEY ou RUNPOD_API_KEY."
            )
    elif endpoint_type == "hunyuan3d":
        key = settings.hunyuan3d_runpod_key or settings.runpod_api_key
        if not key:
            raise RuntimeError(
                "Chave do RunPod para Hunyuan3D-2 não configurada. "
                "Defina HUNYUAN3D_RUNPOD_KEY ou RUNPOD_API_KEY."
            )
    else:
        key = settings.runpod_api_key
        if not key:
            raise RuntimeError("RUNPOD_API_KEY não configurada.")
    
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


async def generate_mesh_instantmesh(image_bytes: bytes) -> bytes:
    """
    Recebe bytes de imagem, chama o endpoint RunPod InstantMesh (legado)
    e retorna os bytes do arquivo .obj gerado.
    
    Mantido para compatibilidade.
    """
    settings = get_settings()

    if not settings.instantmesh_runpod_url:
        raise RuntimeError(
            "INSTANTMESH_RUNPOD_URL não configurado no .env. "
            "Crie o endpoint InstantMesh no RunPod e adicione a URL."
        )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = _headers_for_endpoint("instantmesh")

    async with httpx.AsyncClient(timeout=120) as client:
        # 1. Dispara o job (runsync espera até 90s antes de retornar IN_PROGRESS)
        resp = await client.post(
            settings.instantmesh_runpod_url,
            headers=headers,
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
    return await _poll_job_status(poll_url, job_id, headers)


async def generate_mesh_hunyuan3d(
    image_bytes: bytes,
    format: str = "glb",
    texture: bool = True,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.0,
    octree_resolution: int = 256
) -> bytes:
    """
    Recebe bytes de imagem, chama o endpoint RunPod Hunyuan3D-2
    e retorna os bytes do arquivo 3D gerado (GLB ou OBJ).
    
    Parâmetros:
        image_bytes: Bytes da imagem de entrada
        format: "glb" ou "obj" (default: "glb")
        texture: Se True, gera textura (default: True)
        num_inference_steps: Passos de inferência (default: 100)
        guidance_scale: Escala de guidance (default: 7.0)
        octree_resolution: Resolução da octree (default: 256)
    """
    settings = get_settings()

    if not settings.hunyuan3d_runpod_url:
        raise RuntimeError(
            "HUNYUAN3D_RUNPOD_URL não configurado no .env. "
            "Crie o endpoint Hunyuan3D-2 no RunPod e adicione a URL."
        )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = _headers_for_endpoint("hunyuan3d")

    # Input no formato esperado pelo handler Hunyuan3D-2
    job_input = {
        "input": {
            "image": image_b64,
            "format": format.lower(),
            "texture": texture,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "octree_resolution": octree_resolution
        }
    }

    # Validações
    if format.lower() not in ("glb", "obj"):
        raise ValueError("Formato inválido. Use 'glb' ou 'obj'.")
    
    if num_inference_steps < 1 or num_inference_steps > 200:
        raise ValueError("num_inference_steps deve estar entre 1 e 200.")
    
    if octree_resolution not in (128, 256, 512):
        raise ValueError("octree_resolution deve ser 128, 256 ou 512.")

    async with httpx.AsyncClient(timeout=120) as client:
        # 1. Dispara o job
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

    # 2. Se já terminou (runsync rápido)
    if status == "COMPLETED" and output:
        return _extract_mesh(output)

    # 3. Se ainda está rodando, faz poll
    if not job_id:
        raise RuntimeError(f"RunPod não retornou job ID. Resposta: {data}")

    poll_url = _status_url(settings.hunyuan3d_runpod_url, job_id)
    return await _poll_job_status(poll_url, job_id, headers)


async def generate_mesh(
    image_bytes: bytes,
    use_hunyuan3d: bool = True,
    **hunyuan3d_kwargs
) -> bytes:
    """
    Função principal para geração de mesh.
    
    Por padrão usa Hunyuan3D-2 (worker principal).
    Se use_hunyuan3d=False, usa InstantMesh (legado).
    
    Parâmetros Hunyuan3D-2 podem ser passados como **kwargs.
    """
    if use_hunyuan3d:
        return await generate_mesh_hunyuan3d(image_bytes, **hunyuan3d_kwargs)
    else:
        return await generate_mesh_instantmesh(image_bytes)


# Função de compatibilidade (mantém API existente)
async def generate_mesh_compat(image_bytes: bytes) -> bytes:
    """Compatibilidade com código existente que espera generate_mesh(image_bytes)."""
    return await generate_mesh(image_bytes, use_hunyuan3d=True)