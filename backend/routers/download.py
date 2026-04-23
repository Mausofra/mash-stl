"""
Router: GET /download/{job_id}
Retorna o arquivo .obj gerado para download.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, JSONResponse
from routers.jobs import get_job
from services.mesh import detect_mesh_format, media_type_for_mesh, extension_for_mesh

router = APIRouter()


@router.get("/download/{job_id}")
async def download_mesh(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(404, f"Job '{job_id}' não encontrado.")

    if job["status"] == "processing":
        return JSONResponse({"status": "processing", "message": "Geração ainda em andamento."}, status_code=202)

    if job["status"] == "failed":
        raise HTTPException(500, f"Job falhou: {job.get('error')}")

    mesh_bytes = job.get("mesh_bytes")
    if not mesh_bytes:
        raise HTTPException(500, "Arquivo não disponível.")

    filename = job.get("filename")

    detected_fmt = detect_mesh_format(mesh_bytes)
    media_type = media_type_for_mesh(detected_fmt)

    if not filename:
        filename = f"{job_id}.{extension_for_mesh(detected_fmt)}"

    return Response(
        content=mesh_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
