#!/usr/bin/env python3
"""
Utilitario para RunPod Serverless.

Recursos:
- Lista endpoints serverless existentes.
- Mostra URL runsync pronta para uso no backend.
- Testa conectividade de um endpoint especifico.

Uso:
  python create_endpoint.py
  python create_endpoint.py --name hunyuan3d-2-worker
  python create_endpoint.py --test-id <ENDPOINT_ID>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests

RUNPOD_API_URL = os.getenv("RUNPOD_API_URL", "https://api.runpod.ai/v2")


def get_api_key() -> str:
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY nao configurada no ambiente.")
    return api_key


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def endpoint_runsync_url(endpoint_id: str) -> str:
    return f"{RUNPOD_API_URL}/{endpoint_id}/runsync"


def list_endpoints(headers: dict[str, str]) -> list[dict[str, Any]]:
    resp = requests.get(f"{RUNPOD_API_URL}/serverless/endpoints", headers=headers, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]

    return []


def print_endpoints(endpoints: list[dict[str, Any]], name_filter: str | None = None) -> list[dict[str, Any]]:
    selected = []
    for item in endpoints:
        name = str(item.get("name", ""))
        if name_filter and name_filter.lower() not in name.lower():
            continue
        selected.append(item)

    if not selected:
        print("Nenhum endpoint encontrado com o filtro informado.")
        return []

    print(f"Endpoints encontrados: {len(selected)}")
    for ep in selected:
        ep_id = ep.get("id", "<sem-id>")
        ep_name = ep.get("name", "<sem-nome>")
        print("-")
        print(f"  name: {ep_name}")
        print(f"  id:   {ep_id}")
        print(f"  url:  {endpoint_runsync_url(str(ep_id))}")

    return selected


def test_endpoint(headers: dict[str, str], endpoint_id: str) -> bool:
    url = endpoint_runsync_url(endpoint_id)
    payload = {
        "input": {
            # payload minimo so para validar auth + roteamento; deve falhar no handler com erro de input
            "image": "invalid-base64",
            "format": "glb",
            "texture": False,
            "num_inference_steps": 5,
            "guidance_scale": 7.0,
            "octree_resolution": 128,
        }
    }

    print(f"Testando endpoint: {url}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"HTTP {resp.status_code}")

        if resp.status_code == 200:
            print("Endpoint acessivel. O handler respondeu (mesmo com input invalido).")
            return True

        print("Endpoint nao respondeu com 200. Verifique ID, auth e status do endpoint.")
        try:
            print(resp.text[:500])
        except Exception:
            pass
        return False
    except requests.RequestException as exc:
        print(f"Erro ao chamar endpoint: {exc}")
        return False


def print_manual_create_guide() -> None:
    print("\nComo criar novo serverless endpoint no RunPod via CI:")
    print("1. Configure secrets no GitHub repo:")
    print("   - DOCKERHUB_USERNAME")
    print("   - DOCKERHUB_TOKEN")
    print("   - RUNPOD_DOCKER_REPO (ex.: seu-usuario/hunyuan3d-2-worker)")
    print("2. Faça push em main/master com mudancas em runpod-worker/")
    print("3. Aguarde o workflow build-runpod-worker concluir")
    print("4. No RunPod Console, crie endpoint Custom Container com a imagem publicada")
    print("5. Copie a URL runsync e configure no backend como HUNYUAN3D_RUNPOD_URL")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lista e testa endpoints do RunPod Serverless")
    parser.add_argument("--name", default=None, help="Filtra endpoints por nome")
    parser.add_argument("--test-id", default=None, help="ID do endpoint para teste rapido")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        api_key = get_api_key()
    except RuntimeError as exc:
        print(f"Erro: {exc}")
        print_manual_create_guide()
        return 1

    headers = build_headers(api_key)

    try:
        endpoints = list_endpoints(headers)
    except requests.RequestException as exc:
        print(f"Falha ao listar endpoints: {exc}")
        return 1

    print_endpoints(endpoints, args.name)

    if args.test_id:
        ok = test_endpoint(headers, args.test_id)
        if not ok:
            return 2

    print_manual_create_guide()
    return 0


if __name__ == "__main__":
    sys.exit(main())
