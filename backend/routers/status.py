"""
Router: GET /status/{job_id}
Retorna o progresso do job de geração.
"""
from fastapi import APIRouter, HTTPException
from routers.jobs import get_job

router = APIRouter()


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' não encontrado.")

    return {
        "job_id": job_id,
        "status": job["status"],        # processing | completed | failed
        "progress": job["progress"],    # 0-100
        "error": job.get("error"),
    }
