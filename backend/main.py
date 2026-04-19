import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from routers import generate, status, download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

settings = get_settings()

app = FastAPI(
    title="Mash STL API",
    description="Backend para geração de modelos 3D via InstantMesh + RunPod",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
)

# CORS — permite o frontend React (dev e produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, tags=["Geração 3D"])
app.include_router(status.router,   tags=["Status"])
app.include_router(download.router, tags=["Download"])


@app.get("/health", tags=["Sistema"])
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=settings.debug,
    )
