# Deploy do Worker Hunyuan3D-2 no RunPod

Este documento descreve como construir, testar e implantar o worker Hunyuan3D-2 no RunPod Serverless.

## Visão Geral

O worker implementa o modelo [Hunyuan3D-2 da Tencent](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) para geração de modelos 3D com textura a partir de imagens 2D.

**Características:**
- Shape generation via Hunyuan3D-DiT
- Texture generation via Hunyuan3D-Paint
- Export GLB/OBJ com textura embutida
- Volume persistente para cache de pesos (~15GB)
- Multi-stage Docker build com CUDA

## Pré-requisitos

1. Conta no [RunPod](https://www.runpod.io/) com saldo
2. Chave API do RunPod (`RUNPOD_API_KEY`)
3. Repositório GitHub configurado (para CI/CD)
4. Docker (para build local opcional)

## 1. Configuração do Ambiente

### Variáveis de Ambiente

No arquivo `.env` do backend, adicione:

```bash
# InstantMesh (legado, opcional)
INSTANTMESH_RUNPOD_URL=https://api.runpod.ai/v2/<ID>/runsync
INSTANTMESH_RUNPOD_KEY=rpa_xxxxx

# Hunyuan3D-2 (novo worker principal)
HUNYUAN3D_RUNPOD_URL=https://api.runpod.ai/v2/<ID>/runsync
HUNYUAN3D_RUNPOD_KEY=rpa_xxxxx

# Chave geral (fallback)
RUNPOD_API_KEY=rpa_xxxxx

# Timeouts
RUNPOD_POLL_INTERVAL=5
RUNPOD_MAX_WAIT=1800
```

### Secrets no GitHub (para CI/CD)

Adicione os seguintes secrets no seu repositório GitHub:
- `RUNPOD_API_KEY` - Chave API do RunPod
- `RUNPOD_ENDPOINT_ID` (opcional) - ID do endpoint para build remoto

## 2. Build da Imagem Docker

### Build Local (requer GPU para compilação CUDA)

```bash
cd runpod-worker
docker build -t hunyuan3d-2-worker .
```

**Nota:** O build local pode falhar devido à necessidade de compilar extensões CUDA. Recomendamos usar o build remoto via GitHub Actions.

### Build Remoto via GitHub Actions

1. Push do código para o repositório GitHub
2. O workflow `.github/workflows/build-runpod-worker.yml` será executado automaticamente
3. A imagem será construída e enviada para o registry do RunPod

## 3. Criação do Endpoint no RunPod

### Via Console Web

1. Acesse [RunPod Console](https://www.runpod.io/console/serverless)
2. Clique em "New Endpoint"
3. Configure:
   - **Template**: Custom Container
   - **Container Image**: `registry.runpod.ai/<seu-usuario>/hunyuan3d-2-worker:latest`
   - **Container Disk**: 20GB (mínimo)
   - **GPU**: A100 40GB (recomendado) ou A10/A6000
   - **Max Workers**: 1-2 (dependendo do budget)
   - **Idle Timeout**: 5-10 minutos
   - **Volume**: `/runpod-volume` com **25GB** (recomendado para cache eficiente)
4. Adicione variáveis de ambiente:
   - `PRELOAD_MODELS=true` (pré-carrega modelos no cold start)
   - `MAX_IMAGE_SIZE_MB=5`
   - `MAX_MESH_SIZE_MB=50`
5. Salve e anote a URL do endpoint

### Otimização para Volume de 25GB

Com um volume de 25GB, você tem espaço suficiente para:

1. **Pesos do modelo**: ~15GB (Hunyuan3D-2)
2. **Cache temporário**: ~5GB (arquivos intermediários)
3. **Espaço livre**: ~5GB (para operações seguras)

O handler está otimizado para:
- Verificar espaço em disco antes de cada execução
- Limpar cache antigo automaticamente se espaço ficar abaixo de 5GB
- Usar cache persistente para evitar re-downloads
- Logar uso de espaço para monitoramento

### Via CLI (opcional)

```bash
# Instale o RunPod CLI
pip install runpod

# Configure sua chave API
export RUNPOD_API_KEY="rpa_xxxxx"

# Crie o endpoint
runpodctl endpoint create \
  --name "hunyuan3d-2-worker" \
  --image "registry.runpod.ai/<seu-usuario>/hunyuan3d-2-worker:latest" \
  --gpu-type "NVIDIA A100-SXM4-40GB" \
  --gpu-count 1 \
  --disk-size 20 \
  --idle-timeout 300 \
  --env "PRELOAD_MODELS=true" \
  --volume "/runpod-volume:20"
```

## 4. Testes

### Teste do Handler

```bash
cd runpod-worker
python test_handler.py
```

Para testes mais completos (requer GPU e pesos):
```bash
python test_handler.py --full
```

### Teste da Imagem Docker

```bash
# Build local
docker build -t hunyuan3d-test .

# Executar interativamente
docker run --rm -it --gpus all hunyuan3d-test bash

# Testar handler dentro do container
python /handler.py
```

### Teste da API

Use o arquivo `test_image.png` do diretório raiz:

```bash
# Codificar imagem em base64
python -c "import base64; print(base64.b64encode(open('test_image.png', 'rb').read()).decode()[:100] + '...')"

# Enviar para o endpoint (substitua URL)
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "<base64>",
      "format": "glb",
      "texture": true,
      "num_inference_steps": 50,
      "guidance_scale": 7.0,
      "octree_resolution": 256
    }
  }' \
  "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync"
```

## 5. Integração com o Backend

### Atualização do Backend

O backend já foi atualizado para suportar ambos os workers:

```python
from services.runpod import generate_mesh, generate_mesh_hunyuan3d

# Usar Hunyuan3D-2 (padrão)
mesh_bytes = await generate_mesh(image_bytes)

# Usar InstantMesh (legado)
mesh_bytes = await generate_mesh(image_bytes, use_hunyuan3d=False)

# Parâmetros avançados Hunyuan3D-2
mesh_bytes = await generate_mesh_hunyuan3d(
    image_bytes,
    format="glb",
    texture=True,
    num_inference_steps=100,
    guidance_scale=7.0,
    octree_resolution=256
)
```

### Configuração do Router

O router `backend/routers/generate.py` usa `generate_mesh()` que por padrão utiliza Hunyuan3D-2.

## 6. Monitoramento e Troubleshooting

### Logs do Worker

Acesse os logs no console do RunPod ou via CLI:

```bash
runpodctl endpoint logs <endpoint-id>
```

### Métricas Importantes

1. **Cold Start Time**: 2-5 minutos (download de pesos)
2. **Inference Time**: 30-90 segundos por imagem
3. **VRAM Usage**: ~20GB (A100 40GB recomendado)
4. **Disk Usage**: ~15GB para pesos + 5GB temporário

### Problemas Comuns

#### Cold Start Muito Lento
- Configure `PRELOAD_MODELS=true` no endpoint
- Use volume persistente para cache de pesos
- Considere manter 1 worker sempre ativo

#### Falha na Compilação CUDA
- Use a imagem base `runpod/pytorch` oficial
- Verifique `TORCH_CUDA_ARCH_LIST` no Dockerfile
- Build remoto via GitHub Actions geralmente resolve

#### Erro de Memória GPU
- Reduza `octree_resolution` para 128
- Reduza `num_inference_steps` para 50
- Use GPU com mais VRAM (A100 40GB)

#### Timeout da API
- Aumente `RUNPOD_MAX_WAIT` no .env
- Configure timeout maior no endpoint RunPod

#### Espaço em Disco Insuficiente
- Verifique logs para ver uso atual do volume
- O handler limpa cache automaticamente abaixo de 5GB
- Considere aumentar o volume para 30GB se usar cache extensivo
- Monitore logs: "Espaço em disco em /runpod-volume: X.GB livre de Y.GB total"

## 7. Otimização de Custos

### Estratégias

1. **Volume Persistente**: Cache de pesos reduz cold start
2. **Idle Timeout**: 5-10 minutos balanceia custo/performance
3. **Max Workers**: Limite conforme demanda
4. **GPU Selection**: A10 mais barato que A100 para testes

### Estimativa de Custo

- A100 40GB: ~$1.10/hora por worker
- Cold start: ~3 minutos = $0.055 por cold start
- Inference: ~1 minuto = $0.018 por imagem

**Custo mensal estimado (1000 imagens):**
- 1000 inferências: $18.00
- 50 cold starts: $2.75
- **Total: ~$20-25/mês**

## 8. Rollback para InstantMesh

Caso necessário, volte para InstantMesh:

1. No `.env`, use apenas `INSTANTMESH_RUNPOD_URL`
2. No código, chame `generate_mesh(image_bytes, use_hunyuan3d=False)`
3. Ou configure `HUNYUAN3D_RUNPOD_URL` vazio

## 9. Próximos Passos

1. ✅ Dockerfile multi-stage otimizado
2. ✅ GitHub Actions workflow
3. ✅ Integração com backend
4. ✅ Scripts de teste
5. ✅ Build remoto da imagem
6. 🔄 Criação do endpoint RunPod
7. 🔄 Validação completa
8. 🔄 Monitoramento em produção

### Status Atual (19/04/2026)

✅ **Build da Imagem Concluído:**
- Imagem Docker: `registry.runpod.ai/hunyuan3d-2-worker:7e2a7fe9c43be101e8049d695f35554133ff59891`
- Tag latest: `registry.runpod.ai/hunyuan3d-2-worker:latest`
- Workflow ID: 24645776807 (sucesso)

### Próximo Passo Imediato: Criar Endpoint RunPod

1. **Acesse o Console RunPod:**
   - https://www.runpod.io/console/serverless

2. **Clique em "New Endpoint"**

3. **Configure o Endpoint:**
   - **Template**: Custom Container
   - **Container Image**: `registry.runpod.ai/hunyuan3d-2-worker:latest`
   - **Container Disk**: 20GB
   - **GPU**: A100 40GB (recomendado) ou A100 80GB
   - **Max Workers**: 1
   - **Idle Timeout**: 300 segundos (5 minutos)
   - **Volume**: `/runpod-volume` com **25GB** (CRÍTICO)

4. **Variáveis de Ambiente:**
   - `PRELOAD_MODELS=true` (pré-carrega modelos no startup)
   - `MAX_IMAGE_SIZE_MB=5`
   - `MAX_MESH_SIZE_MB=50`
   - `VOLUME_PATH=/runpod-volume`

5. **Salve e Copie a URL:**
   - URL será: `https://api.runpod.ai/v2/<SEU_ENDPOINT_ID>/runsync`
   - Adicione ao `.env` como `HUNYUAN3D_RUNPOD_URL`

### Script de Ajuda

Execute para verificar endpoints existentes:
```bash
python create_endpoint.py
```

### Após Criar o Endpoint

1. Atualize o `.env` com a URL do endpoint:
   ```bash
   HUNYUAN3D_RUNPOD_URL=https://api.runpod.ai/v2/<SEU_ENDPOINT_ID>/runsync
   ```

2. Teste a integração completa:
   ```bash
   # Inicie o backend
   cd backend
   python -m uvicorn main:app --reload --port 8000

   # Teste via frontend ou API
   ```

3. Monitore os logs no console RunPod para verificar cold start e downloads.

## Suporte

- Issues: [GitHub Repo](https://github.com/Mausofra/mash-stl)
- RunPod: [Documentação](https://docs.runpod.io/)
- Hunyuan3D-2: [Repositório Oficial](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)