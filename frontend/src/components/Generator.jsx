import { useState, useRef, useEffect, useCallback } from 'react'
import { submitGenerate, fetchStatus, downloadUrl } from '../api'
import './Generator.css'

const POLL_MS = 3000

export default function Generator() {
  const [image, setImage]       = useState(null)       // File
  const [preview, setPreview]   = useState(null)       // object URL
  const [dragging, setDragging] = useState(false)

  const [jobId, setJobId]       = useState(null)
  const [status, setStatus]     = useState('idle')     // idle | processing | completed | failed
  const [progress, setProgress] = useState(0)
  const [error, setError]       = useState(null)
  const [meshUrl, setMeshUrl]   = useState(null)

  const fileInputRef  = useRef(null)
  const pollTimerRef  = useRef(null)

  // Limpa o preview anterior ao trocar imagem
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  // Poll de status
  useEffect(() => {
    if (!jobId || status !== 'processing') return

    pollTimerRef.current = setInterval(async () => {
      try {
        const data = await fetchStatus(jobId)
        setProgress(data.progress ?? 0)

        if (data.status === 'completed') {
          clearInterval(pollTimerRef.current)
          setStatus('completed')
          setProgress(100)
          setMeshUrl(downloadUrl(jobId))
        } else if (data.status === 'failed') {
          clearInterval(pollTimerRef.current)
          setStatus('failed')
          setError(data.error || 'Erro desconhecido no servidor.')
        }
      } catch {
        // rede fora temporariamente — ignora e tenta novamente
      }
    }, POLL_MS)

    return () => clearInterval(pollTimerRef.current)
  }, [jobId, status])

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

    try {
      const data = await submitGenerate({ image })
      setJobId(data.job_id)
    } catch (err) {
      setStatus('failed')
      setError(err.message)
    }
  }

  const handleReset = () => {
    clearInterval(pollTimerRef.current)
    setImage(null); setPreview(null)
    setJobId(null); setStatus('idle'); setProgress(0)
    setError(null); setMeshUrl(null)
  }

  return (
    <section className="generator" id="generator">
      <div className="generator-container">
        <div className="generator-header">
          <h2>Gere seu Modelo 3D</h2>
          <p>Envie uma imagem de referência e nossa IA gera o mesh em segundos.</p>
        </div>

        {status === 'idle' || status === 'failed' ? (
          <div className="generator-form">
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
                  <span className="dropzone-hint">PNG, JPG, WebP</span>
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

            {error && <p className="gen-error">⚠️ {error}</p>}

            <button
              className="btn-primary btn-generate"
              onClick={handleSubmit}
              disabled={status === 'processing'}
            >
              Gerar Modelo 3D
            </button>
          </div>
        ) : null}

        {/* Progress */}
        {status === 'processing' && (
          <div className="gen-progress-wrap">
            <p className="gen-status-text">
              {progress < 20 ? '🔍 Analisando imagem...' :
               progress < 60 ? '⚙️ Gerando geometria 3D...' :
               '✨ Finalizando mesh...'}
            </p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <span className="progress-pct">{progress}%</span>
          </div>
        )}

        {/* Resultado */}
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
              <a href={meshUrl} download={`modelo-${jobId}.obj`} className="btn-primary">
                ⬇️ Baixar .OBJ
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
