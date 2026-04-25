"""
Pós-processamento de mesh 3D via Trimesh.

Pipeline (todas as etapas são opcionais via config):
  1. Remove fragmentos isolados — mantém componentes acima do threshold
  2. Corrige normais — recalcula para apontar para fora
  3. Decimation — reduz faces até target_faces (usa fast-simplification)
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def postprocess_mesh(
    mesh_bytes: bytes,
    *,
    remove_fragments: bool = True,
    fragment_threshold: float = 0.01,
    fix_normals: bool = True,
    fill_holes: bool = True,
    smooth: bool = True,
    smooth_iterations: int = 5,
    smooth_lambda: float = 0.5,
    decimate: bool = False,
    target_faces: int = 100_000,
) -> bytes:
    """
    Recebe bytes de um GLB/OBJ, aplica limpeza e retorna bytes do GLB processado.

    fragment_threshold: fração mínima de faces para manter um componente (0.01 = 1%).
    target_faces: número alvo de faces após decimation.
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh não instalado — pós-processamento ignorado.")
        return mesh_bytes

    try:
        scene_or_mesh = trimesh.load(
            io.BytesIO(mesh_bytes),
            file_type="glb",
            force="mesh",
        )
    except Exception as exc:
        logger.warning("Falha ao carregar mesh para pós-processamento: %s", exc)
        return mesh_bytes

    # Normaliza para Trimesh (scene pode conter vários meshes)
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            logger.warning("Scene vazia — pós-processamento ignorado.")
            return mesh_bytes
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    original_faces = len(mesh.faces)
    logger.info("Pós-processamento: mesh com %d faces.", original_faces)

    # 1. Remove fragmentos isolados
    if remove_fragments:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            components = sorted(components, key=lambda c: len(c.faces), reverse=True)
            # Mantém componentes com face_count > threshold * maior_componente
            min_faces = len(components[0].faces) * fragment_threshold
            kept = [components[0]] + [
                c for c in components[1:] if len(c.faces) > min_faces
            ]
            if len(kept) < len(components):
                mesh = trimesh.util.concatenate(kept)
                removed = len(components) - len(kept)
                logger.info("Fragmentos removidos: %d de %d componentes.", removed, len(components))

    # 2. Preenche buracos
    if fill_holes:
        before = len(mesh.faces)
        trimesh.repair.fill_holes(mesh)
        added = len(mesh.faces) - before
        if added:
            logger.info("Buracos preenchidos: +%d faces.", added)

    # 3. Corrige normais
    if fix_normals:
        mesh.fix_normals()
        logger.info("Normais corrigidas.")

    # 4. Suavização Laplaciana (elimina facetamento dos triângulos)
    if smooth:
        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=smooth_lambda,
            iterations=smooth_iterations,
            implicit_time_integration=False,
            volume_constraint=True,
        )
        logger.info("Suavização Laplaciana: %d iterações.", smooth_iterations)

    # 5. Decimation
    if decimate and len(mesh.faces) > target_faces:
        try:
            ratio = target_faces / len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(target_faces)
            logger.info(
                "Decimation: %d → %d faces (ratio %.2f).",
                original_faces, len(mesh.faces), ratio,
            )
        except Exception as exc:
            logger.warning("Decimation falhou: %s — mesh original mantido.", exc)

    logger.info("Pós-processamento concluído: %d faces finais.", len(mesh.faces))

    # Exporta como GLB
    out = io.BytesIO()
    mesh.export(out, file_type="glb")
    return out.getvalue()
