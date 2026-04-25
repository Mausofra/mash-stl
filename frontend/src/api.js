const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Envia imagem (e prompt/quality opcionais) para o backend.
 * Retorna { job_id, status }
 */
export async function submitGenerate({ image, extraImages = [], prompt, quality = 'padrao' }) {
  const form = new FormData()
  if (image)  form.append('image', image)
  if (prompt) form.append('prompt', prompt)
  extraImages.forEach(f => form.append('extra_images', f))
  form.append('quality', quality)

  const res = await fetch(`${API_URL}/generate`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Erro ${res.status}`)
  }
  return res.json()
}

/**
 * Consulta o status de um job.
 * Retorna { job_id, status, progress, error, filename }
 */
export async function fetchStatus(jobId) {
  const res = await fetch(`${API_URL}/status/${jobId}`)
  if (!res.ok) throw new Error(`Erro ao consultar status: ${res.status}`)
  return res.json()
}

/**
 * Retorna a URL de download do mesh gerado.
 */
export function downloadUrl(jobId) {
  return `${API_URL}/download/${jobId}`
}
