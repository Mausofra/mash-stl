"""
Pre-processamento leve de imagem para melhorar consistencia na geracao 3D.
"""
from __future__ import annotations

import io
from PIL import Image, ImageOps


def _safe_bbox_from_alpha(img: Image.Image):
    if img.mode not in ("RGBA", "LA"):
        return None
    alpha = img.getchannel("A")
    return alpha.getbbox()


def _safe_bbox_from_white_bg(img: Image.Image, threshold: int = 245):
    gray = img.convert("L")
    # Pixels muito claros viram fundo (0); restante vira objeto (255).
    mask = gray.point(lambda p: 0 if p > threshold else 255)
    return mask.getbbox()


def _expand_bbox(bbox, width: int, height: int, pad_ratio: float):
    if not bbox:
        return (0, 0, width, height)

    left, top, right, bottom = bbox
    pad_x = int((right - left) * pad_ratio)
    pad_y = int((bottom - top) * pad_ratio)

    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    return (left, top, right, bottom)


def preprocess_image(
    image_bytes: bytes,
    *,
    max_dim: int = 1024,
    autocrop: bool = True,
    pad_ratio: float = 0.08,
    apply_autocontrast: bool = True,
) -> bytes:
    """
    Pipeline leve:
      1) corrige orientacao EXIF
      2) autocrop do objeto (alpha ou fundo branco)
      3) autocontrast leve
      4) resize para max_dim
      5) exporta em PNG
    """
    with Image.open(io.BytesIO(image_bytes)) as src:
        img = ImageOps.exif_transpose(src)

        if autocrop:
            bbox = _safe_bbox_from_alpha(img)
            if bbox is None:
                bbox = _safe_bbox_from_white_bg(img)
            bbox = _expand_bbox(bbox, img.width, img.height, pad_ratio)
            img = img.crop(bbox)

        # Hunyuan recebe RGB; manter pipeline simples para consistencia.
        img = img.convert("RGB")

        if apply_autocontrast:
            img = ImageOps.autocontrast(img, cutoff=1)

        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
