"""
Serviço Ollama — usa qwen2.5vl para descrever uma imagem
e gerar um prompt adequado para geração 3D.
Só é chamado quando o input do usuário é uma imagem.
"""
import base64
import httpx
import logging
from config import get_settings

logger = logging.getLogger(__name__)


async def image_to_prompt(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Envia a imagem para o qwen2.5vl via Ollama e retorna
    uma descrição detalhada para ser usada como prompt 3D.
    """
    settings = get_settings()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": settings.ollama_model_vision,
        "prompt": (
            "Describe this object in detail for 3D model generation. "
            "Focus on: shape, geometry, proportions, materials, surface details, colors. "
            "Be concise and technical. Output only the description, no extra commentary."
        ),
        "images": [image_b64],
        "stream": False,
    }

    url = f"{settings.ollama_url}/api/generate"

    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            prompt = data.get("response", "").strip()
            logger.info("Ollama gerou prompt: %s", prompt[:100])
            return prompt
        except httpx.ConnectError:
            logger.warning("Ollama não disponível. Usando fallback genérico.")
            return "A detailed 3D object based on the provided reference image."
        except Exception as exc:
            logger.error("Erro no Ollama: %s", exc)
            return "A detailed 3D object based on the provided reference image."
