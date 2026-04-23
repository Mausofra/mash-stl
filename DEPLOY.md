# Deploy Serverless no RunPod (Passo a Passo Simples)

Este guia mostra, em linguagem simples, como publicar o worker Hunyuan3D-2 no RunPod usando CI no GitHub.

## Objetivo

Você vai fazer 5 etapas:

1. Configurar os secrets no GitHub.
2. Fazer push para disparar o workflow.
3. Copiar a imagem gerada no GitHub Actions.
4. Criar um endpoint serverless no RunPod com essa imagem.
5. Ligar o backend local nesse endpoint e testar.

## Pré-requisitos

1. Conta no GitHub com acesso ao repositório.
2. Conta no RunPod com saldo.
3. Chave de API do RunPod.
4. Backend deste projeto rodando localmente.

## Etapa 1: Configurar Secrets no GitHub

No repositório do GitHub, vá em Settings > Secrets and variables > Actions e crie:

1. RUNPOD_API_KEY
Valor: sua chave da API do RunPod (começa com rpa_).

2. RUNPOD_DOCKER_REPO
Valor: usuario-ou-org/hunyuan3d-2-worker

Exemplo de valor:
mausofra/hunyuan3d-2-worker

## Etapa 2: Disparar o Build no CI

1. Faça commit das mudanças.
2. Faça push para main ou master.
3. Abra a aba Actions do GitHub.
4. Abra o workflow Build and Deploy RunPod Worker.
5. Aguarde terminar com sucesso.

## Etapa 3: Copiar a Imagem Gerada

Quando o workflow terminar:

1. Entre no job build-and-push.
2. Veja o resumo final (Step Summary).
3. Copie a imagem Latest.

Exemplo:
docker.io/mausofra/hunyuan3d-2-worker:latest

## Etapa 4: Criar Endpoint Serverless no RunPod

1. Acesse o console serverless do RunPod.
2. Clique em New Endpoint.
3. Escolha Custom Container.
4. Em Container Image, cole a imagem copiada da Etapa 3.
5. Configure recursos mínimos:

- GPU: A100 40GB (recomendado).
- Container Disk: 20 GB.
- Volume: /runpod-volume com 25 GB.
- Idle Timeout: 300 segundos.

1. Configure variáveis de ambiente no endpoint:

- PRELOAD_MODELS=true
- MAX_IMAGE_SIZE_MB=5
- MAX_MESH_SIZE_MB=50
- VOLUME_PATH=/runpod-volume

1. Salve o endpoint.
2. Copie a URL runsync do endpoint criado.

Formato da URL:
[URL runsync de exemplo](https://api.runpod.ai/v2/SEU_ENDPOINT_ID/runsync)

## Etapa 5: Ligar Backend ao Endpoint

No arquivo .env da raiz do projeto, preencha:

    HUNYUAN3D_RUNPOD_URL=https://api.runpod.ai/v2/SEU_ENDPOINT_ID/runsync
    HUNYUAN3D_RUNPOD_KEY=sua_chave_rpa
    RUNPOD_API_KEY=sua_chave_rpa
    RUNPOD_MAX_WAIT=1800

## Etapa 6: Testar de Ponta a Ponta

1. Suba o backend.
2. Envie uma imagem pelo frontend.
3. Verifique se o job muda para completed.
4. Faça download do arquivo gerado.

## Comandos úteis

Ver endpoints existentes:
python create_endpoint.py

Filtrar por nome:
python create_endpoint.py --name hunyuan3d-2-worker

Testar endpoint por ID:
python create_endpoint.py --test-id SEU_ENDPOINT_ID

## Problemas comuns

1. Workflow falha no GitHub.
Causa comum: secret faltando.
Verifique RUNPOD_API_KEY e RUNPOD_DOCKER_REPO.

2. Endpoint retorna erro de autenticação.
Causa comum: chave inválida.
Confira HUNYUAN3D_RUNPOD_KEY e RUNPOD_API_KEY.

3. Job demora muito no primeiro teste.
Normal no primeiro cold start.
O worker pode baixar modelos e levar alguns minutos.

4. Erro de espaço em disco.
Aumente o volume para 30 GB.

## Resumo rápido

1. Configurar secrets no GitHub.
2. Fazer push e aguardar Actions.
3. Copiar imagem publicada.
4. Criar endpoint serverless no RunPod.
5. Atualizar .env com URL e chave.
6. Testar pelo frontend.

## Referências

- [RunPod Serverless Console](https://www.runpod.io/console/serverless)
- [Repositório do modelo Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
