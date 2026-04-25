"""
Pre-processamento de imagem para melhorar consistencia na geracao 3D.

Pipeline:
  1) Corrige orientacao EXIF
  2) Remove fundo via rembg (U2Net) — fallback para autocrop heuristico
  3) Autocrop pelo canal alpha
  4) Autocontraste leve
  5) Resize para max_dim
  6) Exporta PNG (RGBA quando rembg ativo, RGB caso contrario)
"""
from __future__ import annotations

import io
import logging
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# rembg é opcional — importado uma vez, sessão reutilizada entre requests
try:
    from rembg import remove as _rembg_remove, new_session as _new_session
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False
    logger.warning("rembg não instalado — usando autocrop heuristico.")

_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = _new_session("u2net")
        logger.info("Sessão rembg (u2net) inicializada.")
    return _rembg_session


def _remove_background(img: Image.Image) -> Image.Image:
    """Remove fundo via rembg e retorna imagem RGBA."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result_bytes = _rembg_remove(buf.getvalue(), session=_get_rembg_session())
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


def _safe_bbox_from_alpha(img: Image.Image):
    if img.mode not in ("RGBA", "LA"):
        return None
    alpha = img.getchannel("A")
    return alpha.getbbox()


def _safe_bbox_from_white_bg(img: Image.Image, threshold: int = 245):
    gray = img.convert("L")
    mask = gray.point(lambda p: 0 if p > threshold else 255)
    return mask.getbbox()


def _expand_bbox(bbox, width: int, height: int, pad_ratio: float):
    if not bbox:
        return (0, 0, width, height)

    left, top, right, bottom = bbox
    pad_x = int((right - left) * pad_ratio)
    pad_y = int((bottom - top) * pad_ratio)

    left   = max(0, left - pad_x)
    top    = max(0, top  - pad_y)
    right  = min(width,  right  + pad_x)
    bottom = min(height, bottom + pad_y)
    return (left, top, right, bottom)


def preprocess_image(
    image_bytes: bytes,
    *,
    max_dim: int = 1024,
    autocrop: bool = True,
    pad_ratio: float = 0.08,
    apply_autocontrast: bool = True,
    use_rembg: bool = True,
) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as src:
        img = ImageOps.exif_transpose(src)

        if use_rembg and _REMBG_AVAILABLE:
            try:
                img = _remove_background(img)
                logger.info("Fundo removido via rembg.")
            except Exception as exc:
                logger.warning("rembg falhou (%s) — usando autocrop heuristico.", exc)
                img = img.convert("RGBA")
        else:
            img = img.convert("RGBA")

        if autocrop:
            bbox = _safe_bbox_from_alpha(img)
            if bbox is None:
                bbox = _safe_bbox_from_white_bg(img)
            bbox = _expand_bbox(bbox, img.width, img.height, pad_ratio)
            img = img.crop(bbox)

        if apply_autocontrast:
            # Autocontraste só nos canais RGB, preserva alpha
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            rgb = ImageOps.autocontrast(rgb, cutoff=1)
            img = Image.merge("RGBA", (*rgb.split(), a))

        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
