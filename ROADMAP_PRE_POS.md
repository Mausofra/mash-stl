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

## Curto prazo (baixo risco, 1 a 2 semanas)

- Perfil de qualidade por requisicao:
  - rapido (menos custo)
  - qualidade (mais passos no worker)
- Validacao de qualidade basica do output:
  - tamanho minimo de mesh
  - retries controlados para falhas transitorias
- Telemetria de pipeline:
  - tempos de pre, inferencia e pos
  - taxa de erro por etapa

## Medio prazo (1080 Ti local + VLM API, 2 a 4 semanas)

- Pre-processamento pesado local:
  - remocao de fundo por modelo dedicado
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
