import { useState, useRef, useEffect, useCallback } from 'react'
import { submitGenerate, fetchStatus, downloadUrl } from '../api'
import './Generator.css'

const POLL_MS     = 3000
const MAX_RETRIES = 1
const MAX_EXTRA   = 3

const QUALITY_OPTIONS = [
  { value: 'rapido', label: 'Rápido',        hint: 'Sem textura · ~1 min' },
  { value: 'padrao', label: 'Padrão',         hint: 'Com textura · ~3 min' },
  { value: 'alta',   label: 'Alta qualidade', hint: 'Com textura · ~5 min' },
]

function progressLabel(progress, quality) {
  if (progress < 15)  return 'Analisando imagem...'
  if (progress < 25)  return 'Enviando para geração...'
  if (progress < 70)  return 'Gerando geometria 3D...'
  if (quality !== 'rapido' && progress < 92) return 'Aplicando texturas PBR...'
  return 'Finalizando mesh...'
}

function MiniUpload({ file, preview, onFile, onRemove, index }) {
  const ref = useRef(null)
  return (
    <div className="mini-upload">
      {preview ? (
        <>
          <img src={preview} alt={`ângulo ${index + 2}`} className="mini-preview" />
          <button className="mini-remove" onClick={onRemove} type="button">✕</button>
        </>
      ) : (
        <div className="mini-placeholder" onClick={() => ref.current?.click()}>
          <span>+</span>
          <span className="mini-label">ângulo {index + 2}</span>
          <input
            ref={ref}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            style={{ display: 'none' }}
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </div>
      )}
    </div>
  )
}

export default function Generator() {
  const [image, setImage]         = useState(null)
  const [preview, setPreview]     = useState(null)
  const [dragging, setDragging]   = useState(false)
  const [quality, setQuality]     = useState('padrao')
  const [prompt, setPrompt]       = useState('')

  // Imagens extras (multi-view)
  const [extras, setExtras]       = useState([null, null, null])
  const [extraPreviews, setExtraPreviews] = useState([null, null, null])

  const [jobId, setJobId]         = useState(null)
  const [status, setStatus]       = useState('idle')
  const [progress, setProgress]   = useState(0)
  const [error, setError]         = useState(null)
  const [meshUrl, setMeshUrl]     = useState(null)
  const [filename, setFilename]   = useState(null)

  const fileInputRef = useRef(null)
  const pollTimerRef = useRef(null)
  const retryRef     = useRef(0)
  const qualityRef   = useRef('padrao')

  useEffect(() => { qualityRef.current = quality }, [quality])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])
  useEffect(() => () => {
    extraPreviews.forEach(p => p && URL.revokeObjectURL(p))
  }, [extraPreviews])

  // Poll de status
  useEffect(() => {
    if (!jobId || status !== 'processing') return

    pollTimerRef.current = setInterval(async () => {
      try {
        const data = await fetchStatus(jobId)
        setProgress(data.progress ?? 0)

        if (data.status === 'completed') {
          clearInterval(pollTimerRef.current)
          retryRef.current = 0
          setStatus('completed')
          setProgress(100)
          setMeshUrl(downloadUrl(jobId))
          setFilename(data.filename || `modelo-${jobId}.glb`)

        } else if (data.status === 'failed') {
          clearInterval(pollTimerRef.current)

          if (retryRef.current < MAX_RETRIES) {
            retryRef.current += 1
            setError(null)
            setProgress(5)
            try {
              const retry = await submitGenerate({
                image,
                extraImages: extras.filter(Boolean),
                prompt,
                quality: qualityRef.current,
              })
              setJobId(retry.job_id)
            } catch {
              retryRef.current = MAX_RETRIES
              setStatus('failed')
              setError('Falha ao reenviar o job.')
            }
          } else {
            retryRef.current = 0
            setStatus('failed')
            setError(data.error || 'Erro desconhecido no servidor.')
          }
        }
      } catch {
        // rede fora temporariamente — ignora
      }
    }, POLL_MS)

    return () => clearInterval(pollTimerRef.current)
  }, [jobId, status, image, extras, prompt])

  const handleFile = useCallback((file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Formato inválido. Use PNG, JPG ou WebP.')
      return
    }
    setError(null)
    setImage(file)
    setPreview(URL.createObjectURL(file))
  }, [])

  const handleExtraFile = useCallback((file, idx) => {
    if (!file || !file.type.startsWith('image/')) return
    setExtras(prev => { const n = [...prev]; n[idx] = file; return n })
    setExtraPreviews(prev => {
      if (prev[idx]) URL.revokeObjectURL(prev[idx])
      const n = [...prev]; n[idx] = URL.createObjectURL(file); return n
    })
  }, [])

  const handleExtraRemove = useCallback((idx) => {
    setExtras(prev => { const n = [...prev]; n[idx] = null; return n })
    setExtraPreviews(prev => {
      if (prev[idx]) URL.revokeObjectURL(prev[idx])
      const n = [...prev]; n[idx] = null; return n
    })
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files?.[0])
  }, [handleFile])

  const handleSubmit = async () => {
    if (!image) { setError('Selecione uma imagem.'); return }
    setError(null)
    setStatus('processing')
    setProgress(5)
    setMeshUrl(null)
    setFilename(null)
    retryRef.current = 0

    try {
      const data = await submitGenerate({
        image,
        extraImages: extras.filter(Boolean),
        prompt: prompt.trim() || undefined,
        quality,
      })
      setJobId(data.job_id)
    } catch (err) {
      setStatus('failed')
      setError(err.message)
    }
  }

  const handleReset = () => {
    clearInterval(pollTimerRef.current)
    retryRef.current = 0
    setImage(null); setPreview(null)
    setExtras([null, null, null]); setExtraPreviews([null, null, null])
    setPrompt('')
    setJobId(null); setStatus('idle'); setProgress(0)
    setError(null); setMeshUrl(null); setFilename(null)
  }

  const downloadLabel = filename
    ? `Baixar .${filename.split('.').pop().toUpperCase()}`
    : 'Baixar modelo'

  const activeExtras = extras.map((f, i) => ({ file: f, preview: extraPreviews[i] }))
  const nextSlot = activeExtras.findIndex(e => !e.file)

  return (
    <section className="generator" id="generator">
      <div className="generator-container">
        <div className="generator-header">
          <h2>Gere seu Modelo 3D</h2>
          <p>Envie uma imagem de referência e nossa IA gera o mesh em segundos.</p>
        </div>

        {(status === 'idle' || status === 'failed') && (
          <div className="generator-form">
            {/* Dropzone principal */}
            <div
              className={`dropzone ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              {preview ? (
                <img src={preview} alt="preview" className="dropzone-preview" />
              ) : (
                <>
                  <span className="dropzone-icon">🖼️</span>
                  <p>Arraste uma imagem ou <strong>clique para selecionar</strong></p>
                  <span className="dropzone-hint">PNG, JPG, WebP · imagem principal</span>
                </>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                style={{ display: 'none' }}
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
            </div>

            {/* Multi-view — ângulos extras */}
            {preview && (
              <div className="multiview-row">
                <span className="multiview-label">Ângulos adicionais <span className="multiview-hint">(opcional · melhora a geometria)</span></span>
                <div className="multiview-slots">
                  {activeExtras.map((e, i) => (
                    (e.file || i === nextSlot) && (
                      <MiniUpload
                        key={i}
                        index={i}
                        file={e.file}
                        preview={e.preview}
                        onFile={(f) => handleExtraFile(f, i)}
                        onRemove={() => handleExtraRemove(i)}
                      />
                    )
                  ))}
                </div>
              </div>
            )}

            {/* Prompt opcional */}
            <div className="prompt-wrap">
              <input
                className="prompt-input"
                type="text"
                placeholder="Descreva o objeto (opcional) — ex: wooden chair with armrests"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                maxLength={200}
              />
            </div>

            {/* Qualidade */}
            <div className="quality-selector">
              {QUALITY_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  className={`quality-btn ${quality === opt.value ? 'active' : ''}`}
                  onClick={() => setQuality(opt.value)}
                  type="button"
                >
                  <span className="quality-label">{opt.label}</span>
                  <span className="quality-hint">{opt.hint}</span>
                </button>
              ))}
            </div>

            {error && <p className="gen-error">⚠️ {error}</p>}

            <button
              className="btn-primary btn-generate"
              onClick={handleSubmit}
              disabled={status === 'processing'}
            >
              Gerar Modelo 3D
            </button>
          </div>
        )}

        {status === 'processing' && (
          <div className="gen-progress-wrap">
            {retryRef.current > 0 && (
              <p className="gen-retry-text">Tentando novamente ({retryRef.current}/{MAX_RETRIES})...</p>
            )}
            <p className="gen-status-text">{progressLabel(progress, quality)}</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <span className="progress-pct">{progress}%</span>
          </div>
        )}

        {status === 'completed' && meshUrl && (
          <div className="gen-result">
            <div className="viewer-wrap">
              <model-viewer
                src={meshUrl}
                alt="Modelo 3D gerado"
                auto-rotate
                camera-controls
                shadow-intensity="1"
                style={{ width: '100%', height: '400px', borderRadius: '12px' }}
              />
            </div>
            <div className="result-actions">
              <a href={meshUrl} download={filename || `modelo-${jobId}.glb`} className="btn-primary">
                ⬇️ {downloadLabel}
              </a>
              <button className="btn-secondary" onClick={handleReset}>
                Gerar outro
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
