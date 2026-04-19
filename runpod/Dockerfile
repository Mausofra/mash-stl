# ── Teste mínimo para validar build RunPod ──
FROM python:3.10-slim

RUN pip install --no-cache-dir runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
