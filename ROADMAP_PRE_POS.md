# Roadmap de Pre e Pos-processamento

## Ja implantado agora

- Pre-processamento leve no backend:
  - correcao de orientacao EXIF
  - autocrop do objeto (alpha/fundo claro)
  - autocontraste leve
  - resize para dimensao maxima configuravel
- Pos-processamento leve no backend:
  - deteccao automatica de formato de mesh (glb/obj/zip)
  - download com MIME type correto
  - nome de arquivo com extensao coerente
- Frontend:
  - perfis de qualidade (rapido / padrao / alta qualidade)
  - progresso granular durante inferencia
  - retry automatico em caso de falha transitoria
  - botao de download com extensao correta (GLB ou OBJ)

## Blocker atual

- OOM (CUDA out of memory) na GPU RTX 4090 (24 GB) ao rodar shape + textura:
  - SHAPE usa ~20 GB, PAINT precisa de ~6 GB adicionais
  - Solucao: migrar para GPU >= 40 GB (A40 48 GB ou A100 40 GB) no RunPod

## Curto prazo (apos resolver OOM)

- Paridade de funcionalidades com site oficial (3d.hunyuan.tencent.com):
  - Prompt de texto no worker:
    - handler aceita campo "prompt" no payload
    - backend passa prompt gerado pelo Ollama ao RunPod
    - SHAPE_PIPELINE recebe prompt para condicionar geracao
  - Multiplas imagens de entrada (multi-view):
    - frontend permite upload de 2 a 4 fotos do mesmo objeto
    - handler passa mv_images ao pipeline
    - resultado geometricamente mais preciso sem "adivinhar" angulos
  - Remocao de fundo automatica (rembg):
    - roda local no backend, sem GPU
    - substitui autocrop simples por segmentacao real do objeto
- Validacao de qualidade basica do output:
  - tamanho minimo de mesh
- Telemetria de pipeline:
  - tempos de pre, inferencia e pos
  - taxa de erro por etapa

## Medio prazo (1080 Ti local + VLM API, 2 a 4 semanas)

- Pre-processamento pesado local:
  - composicao de fundo neutro para estabilizar inferencia
- Enriquecimento por VLM API:
  - descricao estruturada de objeto
  - tags para ajustar parametros de inferencia
- Pos-processamento pesado local:
  - limpeza de malha (fragmentos isolados)
  - correcao de normais
  - decimation controlado (target de faces)

## Longo prazo (produto)

- Versoes de exportacao:
  - web (leve)
  - print/pro (alta qualidade)
- Avaliacao automatica de qualidade:
  - score de geometria
  - score de textura
- Ciclo de melhoria continua:
  - coletar feedback do usuario
  - ajustar perfis e thresholds periodicamente
